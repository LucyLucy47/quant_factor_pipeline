"""
Quantitative Factor Pipeline with GPU acceleration and multiple screening/training options.

This module implements a full workflow for high‑frequency factor discovery, screening,
multi‑frequency fusion, model training and prediction. It unifies the functionality
from the previously separate scripts into a single, self‑contained package, and
introduces several enhancements:

* **GPU 深度优化**：在特征生成与信息系数计算阶段使用 PyTorch 处理 GPU 张量，替代传统 Pandas 运算；
* **非线性筛选**：除了基于 IC 的线性筛选外，还提供基于 LightGBM 的特征重要性筛选，以捕捉因子间非线性组合；
* **多频率融合**：支持将日频因子库（如 Alpha158）与高频因子在模型层面融合，提高策略稳定性；
* **多算法训练**：内置 LightGBM、XGBoost 和图神经网络 (GNN) 三种模型选项，可根据需要灵活选择。

本代码仅用于学术研究，不构成任何投资建议。在运行前请确认数据路径正确，且安装了 PyTorch、LightGBM、XGBoost 等依赖。
"""
from __future__ import annotations

import os
import datetime as dt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import xgboost as xgb


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------

class DataLoader:
    """Load high‑frequency and daily factor data from disk.

    ``data_dir_hf`` 应包含以日期命名的高频 Level‑2 数据文件（CSV 格式）；
    ``data_dir_daily`` 用于日频因子库（如 Alpha158），同样以日期命名。

    用户应根据自己的文件格式修改 ``_parse_hf_file`` 和 ``_parse_daily_file``。
    """

    def __init__(self, data_dir_hf: str, data_dir_daily: str | None = None) -> None:
        self.data_dir_hf = data_dir_hf
        self.data_dir_daily = data_dir_daily

    def update_data(self) -> None:
        """Placeholder for data updating logic (download/synchronize)."""
        print("[DataLoader] update_data called. Implement data downloading here if needed.")

    def _parse_hf_file(self, file_path: str) -> pd.DataFrame:
        """Parse a high‑frequency data file into a DataFrame indexed by (datetime, code).

        The file is expected to contain at least these columns:
        - ``datetime``: timestamp string;
        - ``code``: stock code;
        - ``bid_price1``, ``ask_price1``, ``bid_volume1``, ``ask_volume1``,
          ``last_price``, ``trade_volume``, ``order_imbal``, etc.

        Modify this function if your data format differs.
        """
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index(['datetime', 'code'], inplace=True)
        return df

    def _parse_daily_file(self, file_path: str) -> pd.DataFrame:
        """Parse a daily factor file into a DataFrame indexed by (date, code)."""
        df = pd.read_csv(file_path, index_col=[0, 1])
        # Convert first level to datetime.date if necessary
        if isinstance(df.index.levels[0][0], str):
            df.index = df.index.set_levels(
                [pd.to_datetime(df.index.levels[0]).date, df.index.levels[1]], level=[0, 1]
            )
        df.index.names = ['date', 'code']
        return df

    def load_hf_data(self, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Load high‑frequency data between ``start_date`` and ``end_date`` (inclusive).

        Returns a concatenated DataFrame indexed by (datetime, code).
        """
        frames: List[pd.DataFrame] = []
        current = start_date
        while current <= end_date:
            fname = os.path.join(self.data_dir_hf, f"{current.strftime('%Y%m%d')}.csv")
            if os.path.exists(fname):
                frames.append(self._parse_hf_file(fname))
            current += dt.timedelta(days=1)
        if not frames:
            raise FileNotFoundError("No high‑frequency data files found in the specified range.")
        return pd.concat(frames).sort_index()

    def load_daily_factors(self, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Load daily factor library (e.g., Alpha158) between ``start_date`` and ``end_date``.

        Files are expected to be named ``YYYYMMDD.csv``. If no files are found, returns
        an empty DataFrame.
        """
        if not self.data_dir_daily:
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        current = start_date
        while current <= end_date:
            fpath = os.path.join(self.data_dir_daily, f"{current.strftime('%Y%m%d')}.csv")
            if os.path.exists(fpath):
                frames.append(self._parse_daily_file(fpath))
            current += dt.timedelta(days=1)
        if frames:
            return pd.concat(frames).sort_index()
        else:
            return pd.DataFrame()

    def compute_daily_close(self, hf_data: pd.DataFrame) -> pd.DataFrame:
        """Compute daily closing price for each stock from high‑frequency data."""
        hf_reset = hf_data.reset_index()
        hf_reset['date'] = hf_reset['datetime'].dt.date
        daily_close = hf_reset.groupby(['date', 'code']).last()
        return daily_close[['last_price']]

    def compute_returns(self, daily_close: pd.DataFrame) -> pd.Series:
        """Compute next‑day returns for each (date, code)."""
        daily_close = daily_close.sort_index(level=0)
        grouped = daily_close.groupby(level='code', group_keys=False)['last_price']
        returns = grouped.apply(lambda s: s.pct_change().shift(-1))
        return returns


# ----------------------------------------------------------------------------
# GPU 加速特征生成
# ----------------------------------------------------------------------------

class GPUFeatureGenerator:
    """Generate aggregated high‑frequency features using PyTorch on GPU.

    Features follow the Mask–base feature–aggregation scheme:
    1. Masks: morning (<=09:45), afternoon (>=14:30), above close, below close;
    2. Base features: a list of raw columns (e.g., ``bid_price1``, ``ask_price1``); 
    3. Aggregations: sum, mean, std, max, min.

    The results are returned as a DataFrame indexed by (date, code), with one
    column per generated feature. Computations are performed on GPU if available.
    """

    def __init__(self, base_features: List[str], max_features: int = 5000) -> None:
        self.base_features = base_features
        self.max_features = max_features
        self.masks = [
            ('morning', lambda t: t <= pd.to_datetime('09:45').time()),
            ('afternoon', lambda t: t >= pd.to_datetime('14:30').time()),
        ]
        self.agg_ops = ['sum', 'mean', 'std', 'max', 'min']

    def generate(self, hf_data: pd.DataFrame) -> pd.DataFrame:
        """Generate aggregated features and return a MultiIndex DataFrame."""
        hf_reset = hf_data.reset_index()
        hf_reset['date'] = hf_reset['datetime'].dt.date
        hf_reset['time'] = hf_reset['datetime'].dt.time
        dates = sorted(hf_reset['date'].unique())
        codes = sorted(hf_reset['code'].unique())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Limit combinations to max_features
        combos = []
        for mask_name, _ in self.masks + [('above', None), ('below', None)]:
            for base in self.base_features:
                for op in self.agg_ops:
                    combos.append((mask_name, base, op))
        combos = combos[: self.max_features]
        # Prepare storage
        result: Dict[str, List] = {f"{m}__{b}__{op}": [] for (m, b, op) in combos}
        # Precompute close price map per date and code
        hf_reset['close'] = hf_reset.groupby(['date', 'code'])['last_price'].transform('last')
        # Iterate by date and code
        for date in dates:
            df_day = hf_reset[hf_reset['date'] == date]
            for code in codes:
                df_stock = df_day[df_day['code'] == code]
                if df_stock.empty:
                    for col in result:
                        result[col].append(np.nan)
                    continue
                # Move base features to tensor
                feat_tensor = torch.tensor(df_stock[self.base_features].values, dtype=torch.float32, device=device)
                times_arr = df_stock['time'].values
                close_price = df_stock['close'].iloc[0]
                # Precompute masks
                mask_tensors: Dict[str, torch.Tensor] = {}
                for mask_name, mask_func in self.masks:
                    bool_mask = mask_func(times_arr)
                    mask_tensors[mask_name] = torch.tensor(bool_mask, dtype=torch.bool, device=device)
                mask_tensors['above'] = torch.tensor((df_stock['last_price'].values > close_price), dtype=torch.bool, device=device)
                mask_tensors['below'] = torch.tensor((df_stock['last_price'].values < close_price), dtype=torch.bool, device=device)
                # Compute aggregations
                for (mname, base, op) in combos:
                    m = mask_tensors[mname]
                    vals = feat_tensor[m, self.base_features.index(base)]
                    if vals.numel() == 0:
                        val = np.nan
                    else:
                        if op == 'sum':
                            val = vals.sum().item()
                        elif op == 'mean':
                            val = vals.mean().item()
                        elif op == 'std':
                            val = vals.std(unbiased=False).item()
                        elif op == 'max':
                            val = vals.max().item()
                        elif op == 'min':
                            val = vals.min().item()
                        else:
                            val = np.nan
                    result[f"{mname}__{base}__{op}"].append(val)
        # Build DataFrame
        index = pd.MultiIndex.from_product([dates, codes], names=['date', 'code'])
        return pd.DataFrame(result, index=index)


# ----------------------------------------------------------------------------
# IC 计算与筛选
# ----------------------------------------------------------------------------

def calc_ic_gpu(feature_matrix: torch.Tensor, returns_matrix: torch.Tensor) -> Tuple[float, float]:
    """Compute mean IC and t‑statistic using GPU tensors.

    ``feature_matrix`` and ``returns_matrix`` must have shape (num_dates, num_stocks).
    The function ranks both matrices along the stock dimension and computes their
    Pearson correlation per date (equivalent to Spearman IC).
    """
    device = feature_matrix.device
    def rank_tensor(x: torch.Tensor) -> torch.Tensor:
        tmp = torch.argsort(x, dim=1)
        ranks = torch.argsort(tmp, dim=1).float()
        return ranks
    f_ranks = rank_tensor(feature_matrix)
    r_ranks = rank_tensor(returns_matrix)
    f_center = f_ranks - f_ranks.mean(dim=1, keepdim=True)
    r_center = r_ranks - r_ranks.mean(dim=1, keepdim=True)
    cov = (f_center * r_center).mean(dim=1)
    std_f = f_center.std(dim=1)
    std_r = r_center.std(dim=1)
    ic = cov / (std_f * std_r + 1e-8)
    ic = ic[torch.isfinite(ic)]
    if ic.numel() == 0:
        return float('nan'), float('nan')
    mean_ic = ic.mean().item()
    std_ic = ic.std(unbiased=False).item()
    n = ic.numel()
    t_stat = mean_ic / (std_ic / (n ** 0.5) + 1e-8)
    return mean_ic, t_stat


def evaluate_features_gpu(features_df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Compute mean IC and t‑stat for each feature column using GPU tensors."""
    dates = sorted(list(set(features_df.index.get_level_values(0))))
    codes = sorted(list(set(features_df.index.get_level_values(1))))
    n_dates, n_codes = len(dates), len(codes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Prepare returns tensor
    returns_matrix = torch.zeros((n_dates, n_codes), dtype=torch.float32, device=device)
    for di, d in enumerate(dates):
        y_day = returns.loc[d]
        for ci, c in enumerate(codes):
            returns_matrix[di, ci] = y_day.get(c, np.nan) if c in y_day.index else float('nan')
    returns_matrix = torch.nan_to_num(returns_matrix, nan=0.0)
    stats = {}
    for col in features_df.columns:
        feat_matrix = torch.zeros((n_dates, n_codes), dtype=torch.float32, device=device)
        for di, d in enumerate(dates):
            f_day = features_df.loc[d][col]
            for ci, c in enumerate(codes):
                feat_matrix[di, ci] = f_day.get(c, np.nan) if c in f_day.index else float('nan')
        feat_matrix = torch.nan_to_num(feat_matrix, nan=0.0)
        mean_ic, t_stat = calc_ic_gpu(feat_matrix, returns_matrix)
        stats[col] = {'mean_ic': mean_ic, 't_stat': t_stat}
    return pd.DataFrame(stats).T


def successive_halving_gpu(features_df: pd.DataFrame, returns: pd.Series,
                           num_rounds: int = 3, start_fraction: float = 0.1, threshold: float = 3.0) -> List[str]:
    """Select features using GPU‑accelerated successive halving.

    In each round, a fraction of the earliest dates is used to evaluate features via
    IC t‑statistics; half of the features with the largest |t| are retained. After
    the final round, features with |t| >= ``threshold`` on the full dataset are kept.
    """
    dates = sorted(list(set(features_df.index.get_level_values(0))))
    n_dates = len(dates)
    candidates = list(features_df.columns)
    for r in range(num_rounds):
        frac = min(start_fraction * (2 ** r), 1.0)
        end_idx = max(int(n_dates * frac), 1)
        selected_dates = dates[: end_idx]
        subset_idx = features_df.index.get_level_values(0).isin(selected_dates)
        subset_feats = features_df.loc[subset_idx, candidates]
        subset_returns = returns.loc[subset_idx]
        stats_df = evaluate_features_gpu(subset_feats, subset_returns)
        stats_df['abs_t'] = stats_df['t_stat'].abs()
        keep_n = max(len(stats_df) // 2, 1)
        candidates = stats_df.sort_values('abs_t', ascending=False).head(keep_n).index.tolist()
        print(f"[Halving] Round {r+1}/{num_rounds}: kept {len(candidates)} features on {frac:.2%} data")
        if keep_n <= 1:
            break
    # Final evaluation on full dataset
    final_stats = evaluate_features_gpu(features_df[candidates], returns)
    final_stats['abs_t'] = final_stats['t_stat'].abs()
    survivors = final_stats[final_stats['abs_t'] >= threshold].index.tolist()
    print(f"[Halving] {len(survivors)} features exceed t‑stat threshold {threshold}")
    return survivors


def nonlinear_screening(features_df: pd.DataFrame, returns: pd.Series,
                        num_boost_round: int = 200, importance_threshold: float = 0.0) -> List[str]:
    """Select features based on LightGBM importance to capture non‑linear relations.

    A LightGBM regressor is trained on the entire dataset; features with
    feature importance (gain) greater than ``importance_threshold`` are retained.
    ``importance_threshold`` 默认为 0，表示保留所有正贡献因子。
    """
    params = {
        'objective': 'regression',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'device_type': 'gpu' if torch.cuda.is_available() else 'cpu',
        'metric': 'rmse'
    }
    dataset = lgb.Dataset(features_df, label=returns)
    model = lgb.train(params, dataset, num_boost_round=num_boost_round, verbose_eval=False)
    importances = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    selected = [fn for fn, imp in zip(feature_names, importances) if imp > importance_threshold]
    print(f"[Nonlinear] Selected {len(selected)}/{len(feature_names)} features by LightGBM importance")
    return selected


# ----------------------------------------------------------------------------
# 图神经网络与辅助函数
# ----------------------------------------------------------------------------

class SimpleGCN(nn.Module):
    """A minimal Graph Convolutional Network for per‑stock features."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = adj @ x
        h = self.fc1(h)
        h = self.relu(h)
        h = adj @ h
        h = self.fc2(h)
        return h.squeeze(-1)


def compute_adjacency(stock_features: Dict[str, pd.DataFrame], min_corr: float = 0.1) -> np.ndarray:
    """Compute a normalized adjacency matrix based on absolute Pearson correlations."""
    codes = list(stock_features.keys())
    n = len(codes)
    corr_mat = np.eye(n, dtype=float)
    flattened: List[np.ndarray] = []
    for code in codes:
        df = stock_features[code].dropna()
        flattened.append(df.values.flatten())
    for i in range(n):
        for j in range(i + 1, n):
            if len(flattened[i]) == 0 or len(flattened[j]) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(flattened[i], flattened[j])[0, 1]
            if abs(corr) >= min_corr:
                corr_mat[i, j] = corr_mat[j, i] = abs(corr)
    adj = corr_mat + np.eye(n)
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    return adj_norm.astype(np.float32)


def prepare_gnn_data(features: pd.DataFrame, returns: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, pd.DataFrame]]:
    """Pivot feature DataFrame into 3D array for GNN and compute stock‑wise feature history."""
    dates = sorted(list(set(features.index.get_level_values(0))))
    codes = sorted(list(set(features.index.get_level_values(1))))
    n_dates = len(dates)
    n_codes = len(codes)
    n_feats = features.shape[1]
    X_arr = np.zeros((n_dates, n_codes, n_feats), dtype=np.float32)
    y_arr = np.zeros((n_dates, n_codes), dtype=np.float32)
    stock_feats: Dict[str, pd.DataFrame] = {c: pd.DataFrame(index=dates, columns=features.columns) for c in codes}
    for di, d in enumerate(dates):
        df_day = features.loc[d]
        ret_day = returns.loc[d]
        for ci, c in enumerate(codes):
            if c in df_day.index:
                X_arr[di, ci, :] = df_day.loc[c].values.astype(np.float32)
                y_arr[di, ci] = ret_day.loc[c]
                stock_feats[c].loc[d] = df_day.loc[c].values
    return X_arr, y_arr, codes, stock_feats


def train_gnn_model(X: np.ndarray, y: np.ndarray, adj: np.ndarray, epochs: int = 50,
                    lr: float = 1e-3, verbose: bool = True) -> nn.Module:
    """Train a simple GCN on the provided feature and label arrays."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_stocks, num_feats = X.shape[1], X.shape[2]
    model = SimpleGCN(in_dim=num_feats).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(X.shape[0]):
            x_t = torch.tensor(X[i], dtype=torch.float32, device=device)
            y_t = torch.tensor(y[i], dtype=torch.float32, device=device)
            opt.zero_grad()
            preds = model(x_t, adj_t)
            loss = loss_fn(preds, y_t)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"[GNN] Epoch {epoch+1}/{epochs}, loss={total_loss / X.shape[0]:.6f}")
    return model


def predict_gnn(model: nn.Module, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Use a trained GCN to generate predictions for each sample."""
    device = next(model.parameters()).device
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    preds_list: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(X.shape[0]):
            x_t = torch.tensor(X[i], dtype=torch.float32, device=device)
            preds = model(x_t, adj_t).cpu().numpy()
            preds_list.append(preds)
    return np.stack(preds_list, axis=0)


# ----------------------------------------------------------------------------
# 多频率融合
# ----------------------------------------------------------------------------

def fuse_factors(hf_features: pd.DataFrame, daily_factors: pd.DataFrame) -> pd.DataFrame:
    """Concatenate high‑frequency features with daily factors along columns."""
    return pd.concat([hf_features, daily_factors], axis=1).sort_index()


# ----------------------------------------------------------------------------
# 模型训练接口
# ----------------------------------------------------------------------------

def train_model(features_df: pd.DataFrame, returns: pd.Series, algorithm: str,
                train_dates: Tuple[str, str], valid_dates: Tuple[str, str], test_dates: Tuple[str, str],
                xgb_params: Dict | None = None, gnn_epochs: int = 50) -> Tuple[pd.Series, Dict]:
    """Train a regression model (LightGBM, XGBoost or GNN) and return test predictions.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix indexed by (date, code).
    returns : pd.Series
        Next‑day returns indexed by (date, code).
    algorithm : str
        'lightgbm', 'xgboost', or 'gnn'.
    train_dates, valid_dates, test_dates : tuple
        Date ranges for splitting the dataset.
    xgb_params : dict, optional
        Additional parameters for XGBoost.
    gnn_epochs : int, optional
        Epochs for GNN training.

    Returns
    -------
    pd.Series
        Predictions for the test period indexed by (date, code).
    dict
        Metadata including trained model.
    """
    idx_dates = features_df.index.get_level_values(0)
    m_train = (idx_dates >= train_dates[0]) & (idx_dates <= train_dates[1])
    m_valid = (idx_dates >= valid_dates[0]) & (idx_dates <= valid_dates[1])
    m_test = (idx_dates >= test_dates[0]) & (idx_dates <= test_dates[1])
    X_train = features_df.loc[m_train]
    y_train = returns.loc[m_train]
    X_valid = features_df.loc[m_valid]
    y_valid = returns.loc[m_valid]
    X_test = features_df.loc[m_test]
    y_test = returns.loc[m_test]
    meta: Dict = {}
    if algorithm == 'lightgbm':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'device_type': 'gpu' if torch.cuda.is_available() else 'cpu'
        }
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        model = lgb.train(params, lgb_train, num_boost_round=300,
                          valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'],
                          early_stopping_rounds=50, verbose_eval=False)
        preds = pd.Series(model.predict(X_test, num_iteration=model.best_iteration), index=X_test.index)
        meta['model'] = model
        return preds, meta
    elif algorithm == 'xgboost':
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(xgb_params, dtrain, num_boost_round=300, evals=watchlist,
                          early_stopping_rounds=50, verbose_eval=False)
        preds = pd.Series(model.predict(dtest, iteration_range=(0, model.best_iteration + 1)), index=X_test.index)
        meta['model'] = model
        return preds, meta
    elif algorithm == 'gnn':
        X_arr, y_arr, codes, stock_feats = prepare_gnn_data(features_df, returns)
        adj = compute_adjacency(stock_feats)
        model = train_gnn_model(X_arr, y_arr, adj, epochs=gnn_epochs)
        preds_arr = predict_gnn(model, X_arr, adj)
        dates_unique = sorted(list(set(features_df.index.get_level_values(0))))
        preds_series = pd.Series(index=features_df.index, dtype=float)
        for i, d in enumerate(dates_unique):
            for j, c in enumerate(codes):
                preds_series.loc[(d, c)] = preds_arr[i, j]
        meta['model'] = model
        return preds_series.loc[X_test.index], meta
    else:
        raise ValueError("Unsupported algorithm: choose from 'lightgbm', 'xgboost', 'gnn'.")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def run_pipeline(data_dir_hf: str, data_dir_daily: str | None, start_date: str, end_date: str,
                 train_range: Tuple[str, str], valid_range: Tuple[str, str], test_range: Tuple[str, str],
                 base_features: List[str], screening: str = 'linear', algorithm: str = 'lightgbm') -> pd.Series:
    """Execute the full pipeline: load data, generate features, screen, fuse and train.

    Parameters
    ----------
    data_dir_hf : str
        Directory containing high‑frequency CSV files.
    data_dir_daily : str or None
        Directory containing daily factor files (optional).
    start_date, end_date : str
        Date range for loading data (inclusive, format 'YYYY‑MM‑DD').
    train_range, valid_range, test_range : tuple of str
        Date ranges for splitting the data.
    base_features : list of str
        Raw columns from high‑frequency data used to generate aggregated features.
    screening : str, optional
        Feature screening method: 'linear' (successive halving) or 'nonlinear' (LightGBM importance).
    algorithm : str, optional
        Model training algorithm: 'lightgbm', 'xgboost' or 'gnn'.

    Returns
    -------
    pd.Series
        Predictions for the test period indexed by (date, code).
    """
    # Convert dates
    start = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    end = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    # Load data
    loader = DataLoader(data_dir_hf, data_dir_daily)
    loader.update_data()
    hf_data = loader.load_hf_data(start, end)
    # Feature generation
    gen = GPUFeatureGenerator(base_features=base_features)
    print("[Pipeline] Generating high‑frequency features …")
    hf_feats = gen.generate(hf_data)
    # Compute returns
    daily_close = loader.compute_daily_close(hf_data)
    returns = loader.compute_returns(daily_close)
    # Align indices
    idx = hf_feats.index.intersection(returns.dropna().index)
    hf_feats = hf_feats.loc[idx]
    returns = returns.loc[idx]
    # Screening
    if screening == 'linear':
        selected_cols = successive_halving_gpu(hf_feats, returns, num_rounds=3, start_fraction=0.1, threshold=3.0)
    elif screening == 'nonlinear':
        # To reduce computational burden, perform a preliminary halving to limit candidates
        prelim_cols = successive_halving_gpu(hf_feats, returns, num_rounds=2, start_fraction=0.1, threshold=0.0)
        selected_cols = nonlinear_screening(hf_feats[prelim_cols], returns)
    else:
        raise ValueError("screening must be 'linear' or 'nonlinear'")
    hf_selected = hf_feats[selected_cols]
    # Load daily factors
    if data_dir_daily:
        daily_factors = loader.load_daily_factors(start, end)
        daily_factors = daily_factors.reindex(hf_selected.index).dropna()
        combined = fuse_factors(hf_selected, daily_factors)
    else:
        combined = hf_selected
    # Train and predict
    preds, meta = train_model(combined, returns, algorithm, train_range, valid_range, test_range)
    print("[Pipeline] Finished. Returning test predictions.")
    return preds


if __name__ == '__main__':
    # Example usage (fill in your data paths and date ranges)
    DATA_DIR_HF = './data/high_freq'
    DATA_DIR_DAILY = './data/daily_factors'  # Set to None if no daily factors
    START_DATE = '2016-01-01'
    END_DATE = '2024-12-31'
    TRAIN_RANGE = ('2016-01-01', '2019-12-31')
    VALID_RANGE = ('2020-01-01', '2021-12-31')
    TEST_RANGE = ('2022-01-01', '2024-12-31')
    BASE_FEATURES = ['bid_price1', 'ask_price1', 'bid_volume1', 'ask_volume1',
                     'last_price', 'trade_volume', 'order_imbal']
    # Run pipeline with linear screening and LightGBM
    run_pipeline(DATA_DIR_HF, DATA_DIR_DAILY, START_DATE, END_DATE,
                 TRAIN_RANGE, VALID_RANGE, TEST_RANGE,
                 base_features=BASE_FEATURES,
                 screening='linear',
                 algorithm='lightgbm')