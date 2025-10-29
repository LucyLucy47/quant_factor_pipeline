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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

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
                           num_rounds: int = 3, start_fraction: float = 0.1,
                           threshold: float = 3.0, min_survivors: int = 5000) -> List[str]:
    """Select features using GPU‑accelerated successive halving.

    In each round, a fraction of the earliest dates is used to evaluate features via
    IC t‑statistics; half of the features with the largest |t| are retained. After
    the final round, features with |t| >= ``threshold`` on the full dataset are kept.
    To ensure sufficient diversity, if fewer than ``min_survivors`` survive the
    threshold, the top ``min_survivors`` features by absolute t‑stat are selected.

    Parameters
    ----------
    features_df : pd.DataFrame
        Candidate factor values indexed by (date, code).
    returns : pd.Series
        Next‑day returns indexed by (date, code).
    num_rounds : int
        Number of halving rounds.
    start_fraction : float
        Fraction of the earliest dates to use in the first round; doubles each round.
    threshold : float
        Absolute t‑stat threshold for the final selection.
    min_survivors : int
        Minimum number of features to retain.  If the number of survivors
        exceeding ``threshold`` is less than this, the top ``min_survivors`` by
        |t| are returned.

    Returns
    -------
    List[str]
        List of selected feature names.
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
        # Stop early if number of candidates is already below twice the minimum survivors
        if len(candidates) <= max(min_survivors * 2, 1):
            break
    # Final evaluation on full dataset
    final_stats = evaluate_features_gpu(features_df[candidates], returns)
    final_stats['abs_t'] = final_stats['t_stat'].abs()
    survivors = final_stats[final_stats['abs_t'] >= threshold].index.tolist()
    # Ensure at least min_survivors features
    if len(survivors) < min_survivors:
        survivors = final_stats.sort_values('abs_t', ascending=False).head(min_survivors).index.tolist()
    print(f"[Halving] {len(survivors)} features retained (threshold {threshold}, min {min_survivors})")
    return survivors


def polyphonic_screening(
    features_df: pd.DataFrame,
    returns: pd.Series,
    min_survivors: int = 5000,
    n_clusters: int | None = None,
    segments: int = 5,
) -> List[str]:
    """
    Enhanced multi‑segment (“polyphonic”) screening that preserves diverse factor behaviours.

    Inspired by the OpenFE论文中的“扩张–缩减–筛选”框架【499460586691068†L160-L200】、
    高频因子 IC 连续二分筛选与多样性保持的讨论【499460586691068†L340-L369】,
    此函数对因子在多个时间段上的表现进行分析。步骤如下：

    1. 将整段时间划分为 ``segments`` 个非重叠子区间，并在每个子区间计算每个因子的 t 统计量。
    2. 计算全样本上的 t 统计、各子区间绝对 t 值的平均值 ``mean_abs_t`` 和标准差 ``std_t``，以衡量因子的整体强度和稳定性。
    3. 使用 K‑means 聚类基于子区间 t 值向量将因子分群，聚类数默认为 ``max(2, min_survivors // 200)``，
       使得每个簇含若干百个因子。
    4. 为每个簇分配保留名额 ``k``，其权重由簇中因子的 ``score`` 总和决定，
       其中 ``score = |full_t| + mean_abs_t - std_t``，鼓励整体表现强且跨段稳定的因子。
    5. 在每个簇中按 ``score`` 降序选择 ``k`` 个因子；若总数不足 ``min_survivors``，再从剩余因子中按 ``score`` 最高填充。

    该算法旨在在保证最终保留因子数不少于 ``min_survivors`` 的前提下，充分考虑不同市场阶段的差异，
    平衡强度与稳定性，筛选出更具多样性的因子集合。

    Parameters
    ----------
    features_df : pd.DataFrame
        候选因子矩阵，索引为 (date, code)。
    returns : pd.Series
        下一日收益序列，索引为 (date, code)。
    min_survivors : int, optional
        最少要保留的因子数量，默认为 5000。
    n_clusters : int or None, optional
        聚类簇数量。如果为 None，则设为 ``max(2, min_survivors // 200)``。
    segments : int, optional
        时间段划分的数量，默认为 5。

    Returns
    -------
    List[str]
        选出的因子名列表。
    """
    dates = sorted(list(set(features_df.index.get_level_values(0))))
    n_dates = len(dates)
    segments = max(1, int(segments))
    # 计算各段边界
    bounds = [int(n_dates * (i / segments)) for i in range(1, segments)]
    slices: List[List] = []
    prev = 0
    for b in bounds:
        slices.append(dates[prev:b])
        prev = b
    slices.append(dates[prev:])
    # 按时间段计算 t 统计
    stats_segments: List[pd.Series] = []
    for seg_dates in slices:
        seg_idx = features_df.index.get_level_values(0).isin(seg_dates)
        seg_feats = features_df.loc[seg_idx]
        seg_returns = returns.loc[seg_idx]
        seg_stats = evaluate_features_gpu(seg_feats, seg_returns)
        stats_segments.append(seg_stats['t_stat'])
    # 全样本统计
    full_stats = evaluate_features_gpu(features_df, returns)
    # 组装 DataFrame 并计算 stability 指标
    stat_df = pd.DataFrame({f"t{i+1}": stats_segments[i] for i in range(len(stats_segments))})
    stat_df['full_t'] = full_stats['t_stat']
    # 均值与标准差
    stat_df['mean_abs_t'] = stat_df[[c for c in stat_df.columns if c.startswith('t')]].abs().mean(axis=1)
    stat_df['std_t'] = stat_df[[c for c in stat_df.columns if c.startswith('t')]].std(axis=1)
    stat_df['abs_full_t'] = stat_df['full_t'].abs()
    # score: 强度 + 稳定性
    stat_df['score'] = stat_df['abs_full_t'] + stat_df['mean_abs_t'] - stat_df['std_t']
    stat_df = stat_df.fillna(0.0)
    # 聚类
    if n_clusters is None:
        n_clusters = max(2, min_survivors // 200)
    try:
        km = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = km.fit_predict(stat_df[[c for c in stat_df.columns if c.startswith('t')]])
    except Exception:
        cluster_labels = np.zeros(len(stat_df), dtype=int)
    stat_df['cluster'] = cluster_labels
    # 根据簇内 score 总和分配名额
    survivors: List[str] = []
    total_score_per_cluster: Dict[int, float] = stat_df.groupby('cluster')['score'].sum().to_dict()
    total_score_all = sum(total_score_per_cluster.values())
    for cluster_id, total_score in total_score_per_cluster.items():
        if total_score_all == 0:
            k = max(1, int(len(stat_df[stat_df['cluster'] == cluster_id]) / len(stat_df) * min_survivors))
        else:
            k = max(1, int(total_score / total_score_all * min_survivors))
        cluster_stats = stat_df[stat_df['cluster'] == cluster_id]
        selected = cluster_stats.sort_values('score', ascending=False).head(k).index.tolist()
        survivors.extend(selected)
    # 如数量不足，按 score 补充
    if len(survivors) < min_survivors:
        remaining = stat_df.loc[~stat_df.index.isin(survivors)].sort_values('score', ascending=False)
        survivors.extend(remaining.index.tolist()[: (min_survivors - len(survivors))])
    # 截断到 min_survivors
    survivors = survivors[:min_survivors]
    print(f"[Polyphonic] Retained {len(survivors)} features across {n_clusters} clusters (segments={segments})")
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
    """Train a simple GCN on the provided feature and label arrays with optional multi‑GPU support.

    If more than one CUDA device is available, the model will be wrapped in
    ``torch.nn.DataParallel`` to leverage all GPUs.  This provides data
    parallelism over the batch dimension (dates), improving utilisation when
    training on large datasets.  When only a single GPU is present, the
    behaviour remains unchanged.
    """
    # Determine device and whether to use multiple GPUs
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    num_stocks, num_feats = X.shape[1], X.shape[2]
    model = SimpleGCN(in_dim=num_feats)
    # If multiple GPUs are available, wrap the model for data parallelism
    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # Move adjacency matrix to device once
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # Iterate over each day as a separate sample
        for i in range(X.shape[0]):
            x_t = torch.tensor(X[i], dtype=torch.float32, device=device)
            y_t = torch.tensor(y[i], dtype=torch.float32, device=device)
            opt.zero_grad()
            preds = model(x_t, adj_t)  # DataParallel will split x_t across GPUs if applicable
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
        Name of the regression algorithm. Supported values are:
        'lightgbm', 'xgboost', 'gnn', 'random_forest', 'ridge', and 'svr'.
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
    elif algorithm == 'random_forest':
        # Simple hyperparameter tuning on validation set
        param_grid = [
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 200, 'max_depth': None},
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 10},
        ]
        best_score = -np.inf
        best_params: Dict | None = None
        for params in param_grid:
            model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'],
                                          n_jobs=-1, random_state=0)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_valid)
            score = -mean_squared_error(y_valid, preds_val)
            if score > best_score:
                best_score = score
                best_params = params
        # Train final model on train + valid
        assert best_params is not None
        model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                      max_depth=best_params['max_depth'],
                                      n_jobs=-1, random_state=0)
        model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
        preds = pd.Series(model.predict(X_test), index=X_test.index)
        meta['model'] = model
        meta['best_params'] = best_params
        return preds, meta
    elif algorithm == 'ridge':
        alphas = [0.1, 1.0, 10.0]
        best_score = -np.inf
        best_alpha = None
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_valid)
            score = -mean_squared_error(y_valid, preds_val)
            if score > best_score:
                best_score = score
                best_alpha = alpha
        assert best_alpha is not None
        # Final model
        model = Ridge(alpha=best_alpha)
        model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
        preds = pd.Series(model.predict(X_test), index=X_test.index)
        meta['model'] = model
        meta['best_alpha'] = best_alpha
        return preds, meta
    elif algorithm == 'svr':
        # Standardize features for SVR
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_valid_s = scaler.transform(X_valid)
        X_test_s = scaler.transform(X_test)
        param_grid = [
            {'C': 0.1, 'epsilon': 0.01},
            {'C': 1.0, 'epsilon': 0.01},
            {'C': 10.0, 'epsilon': 0.01},
            {'C': 0.1, 'epsilon': 0.1},
            {'C': 1.0, 'epsilon': 0.1},
            {'C': 10.0, 'epsilon': 0.1},
        ]
        best_score = -np.inf
        best_params: Dict | None = None
        for params in param_grid:
            model = SVR(C=params['C'], epsilon=params['epsilon'], kernel='rbf')
            model.fit(X_train_s, y_train)
            preds_val = model.predict(X_valid_s)
            score = -mean_squared_error(y_valid, preds_val)
            if score > best_score:
                best_score = score
                best_params = params
        assert best_params is not None
        # Final model
        model = SVR(C=best_params['C'], epsilon=best_params['epsilon'], kernel='rbf')
        model.fit(np.vstack([X_train_s, X_valid_s]), np.concatenate([y_train.values, y_valid.values]))
        preds = pd.Series(model.predict(X_test_s), index=X_test.index)
        meta['model'] = model
        meta['best_params'] = best_params
        return preds, meta
    elif algorithm == 'elastic_net':
        # Tune alpha and l1_ratio
        param_grid = [
            {'alpha': 0.1, 'l1_ratio': 0.1},
            {'alpha': 0.1, 'l1_ratio': 0.5},
            {'alpha': 0.1, 'l1_ratio': 0.9},
            {'alpha': 1.0, 'l1_ratio': 0.1},
            {'alpha': 1.0, 'l1_ratio': 0.5},
            {'alpha': 1.0, 'l1_ratio': 0.9},
        ]
        best_score = -np.inf
        best_params: Dict | None = None
        for params in param_grid:
            model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=1000)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_valid)
            score = -mean_squared_error(y_valid, preds_val)
            if score > best_score:
                best_score = score
                best_params = params
        assert best_params is not None
        model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'], max_iter=1000)
        model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
        preds = pd.Series(model.predict(X_test), index=X_test.index)
        meta['model'] = model
        meta['best_params'] = best_params
        return preds, meta
    elif algorithm == 'lasso':
        alphas = [0.01, 0.1, 1.0, 10.0]
        best_score = -np.inf
        best_alpha = None
        for alpha in alphas:
            model = Lasso(alpha=alpha, max_iter=1000)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_valid)
            score = -mean_squared_error(y_valid, preds_val)
            if score > best_score:
                best_score = score
                best_alpha = alpha
        assert best_alpha is not None
        model = Lasso(alpha=best_alpha, max_iter=1000)
        model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
        preds = pd.Series(model.predict(X_test), index=X_test.index)
        meta['model'] = model
        meta['best_alpha'] = best_alpha
        return preds, meta
    elif algorithm == 'mlp':
        # Simple MLP with one hidden layer; tune hidden sizes and alpha
        param_grid = [
            {'hidden_layer_sizes': (64,), 'alpha': 1e-4},
            {'hidden_layer_sizes': (128,), 'alpha': 1e-4},
            {'hidden_layer_sizes': (64, 32), 'alpha': 1e-5},
        ]
        best_score = -np.inf
        best_params: Dict | None = None
        for params in param_grid:
            model = MLPRegressor(hidden_layer_sizes=params['hidden_layer_sizes'], alpha=params['alpha'],
                                 max_iter=200, random_state=0)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_valid)
            score = -mean_squared_error(y_valid, preds_val)
            if score > best_score:
                best_score = score
                best_params = params
        assert best_params is not None
        model = MLPRegressor(hidden_layer_sizes=best_params['hidden_layer_sizes'], alpha=best_params['alpha'],
                             max_iter=200, random_state=0)
        model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
        preds = pd.Series(model.predict(X_test), index=X_test.index)
        meta['model'] = model
        meta['best_params'] = best_params
        return preds, meta
    else:
        raise ValueError(
            "Unsupported algorithm: choose from 'lightgbm', 'xgboost', 'gnn', 'random_forest', 'ridge', 'svr', 'elastic_net', 'lasso', or 'mlp'."
        )


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
        Feature screening method: 'linear' (successive halving),
        'nonlinear' (LightGBM importance) or 'polyphonic' (multi‑segment clustering).  In
        the polyphonic mode, t‑statistics are computed over multiple time segments and
        features are clustered to ensure diversity while retaining a minimum number
        of survivors (default 5000).
    algorithm : str, optional
        Model training algorithm. Supported values are 'lightgbm', 'xgboost',
        'gnn', 'random_forest', 'ridge' and 'svr'.

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
    elif screening == 'polyphonic':
        # Use multi‑segment clustering based on t‑statistics to ensure diversity
        selected_cols = polyphonic_screening(hf_feats, returns, min_survivors=5000)
    else:
        raise ValueError("screening must be 'linear', 'nonlinear' or 'polyphonic'")
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