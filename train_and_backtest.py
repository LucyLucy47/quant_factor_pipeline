"""
Train multiple models on factor data and perform a simple backtest.

This script bridges the outputs of the factor discovery pipeline and the
model evaluation stage.  Given a feature matrix and corresponding
returns, it trains a selected algorithm (LightGBM, XGBoost, GNN or other
classical models) to generate out‑of‑sample predictions and then
constructs a daily long portfolio by selecting stocks with the highest
predicted returns.  Supported algorithms include gradient boosting
models (LightGBM, XGBoost), a graph convolutional network (GNN), and
classical machine learning models (Random Forest, Ridge Regression and
Support Vector Regression).  The portfolio performance is summarised
with cumulative returns, annualised return, Sharpe ratio and maximum
drawdown.

Usage:
    python train_and_backtest.py \
        --features_path data/selected_features.csv \
        --returns_path data/returns.csv \
        --train_start 2016-01-01 --train_end 2019-12-31 \
        --valid_start 2020-01-01 --valid_end 2021-12-31 \
        --test_start  2022-01-01 --test_end  2024-12-31 \
        --algorithm lightgbm --topk 10

Dependencies:
    - numpy
    - pandas
    - torch
    - lightgbm
    - xgboost
    - quant_factor_pipeline (must be in Python path)

Note:
    This backtesting implementation is intentionally simple and serves
    only as an educational example.  It does not account for
    transaction costs, slippage, position limits or risk management.
    Please do not use it directly for live trading without proper
    adaptation.
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from quant_factor_pipeline import train_model


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train a model on factor data and backtest a simple strategy.")
    parser.add_argument("--features_path", type=str, required=True,
                        help="CSV file containing features with a MultiIndex (date, code).")
    parser.add_argument("--returns_path", type=str, required=True,
                        help="CSV file containing next‑day returns with a MultiIndex (date, code).")
    parser.add_argument("--train_start", type=str, required=True, help="Training start date (YYYY-MM-DD).")
    parser.add_argument("--train_end", type=str, required=True, help="Training end date (YYYY-MM-DD).")
    parser.add_argument("--valid_start", type=str, required=True, help="Validation start date (YYYY-MM-DD).")
    parser.add_argument("--valid_end", type=str, required=True, help="Validation end date (YYYY-MM-DD).")
    parser.add_argument("--test_start", type=str, required=True, help="Test start date (YYYY-MM-DD).")
    parser.add_argument("--test_end", type=str, required=True, help="Test end date (YYYY-MM-DD).")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lightgbm",
        choices=[
            "lightgbm",
            "xgboost",
            "gnn",
            "random_forest",
            "ridge",
            "svr",
            "elastic_net",
            "lasso",
            "mlp",
        ],
        help=(
            "Regression algorithm to use. Supported values: lightgbm, xgboost, gnn, "
            "random_forest, ridge, svr, elastic_net, lasso, mlp."
        ),
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of top stocks to hold each day in the backtest.")
    return parser.parse_args()


def load_data(features_path: str, returns_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and returns from CSV files.

    The CSVs must have a MultiIndex of the form (date, code).  The date
    column should parseable to datetime.date.  The returns file must
    contain a column named 'return'.
    """
    # Load features
    feats = pd.read_csv(features_path, index_col=[0, 1])
    # Convert string date index to date
    feats.index = pd.MultiIndex.from_tuples(
        [(dt.datetime.strptime(d, "%Y-%m-%d").date() if isinstance(d, str) else d, c)
         for d, c in feats.index],
        names=["date", "code"]
    )
    # Load returns
    rets_df = pd.read_csv(returns_path, index_col=[0, 1])
    rets_df.index = pd.MultiIndex.from_tuples(
        [(dt.datetime.strptime(d, "%Y-%m-%d").date() if isinstance(d, str) else d, c)
         for d, c in rets_df.index],
        names=["date", "code"]
    )
    if 'return' in rets_df.columns:
        returns = rets_df['return']
    else:
        # If the CSV has a single column without a name, treat it as returns
        returns = rets_df.iloc[:, 0]
    return feats, returns


def backtest(preds: pd.Series, returns: pd.Series, topk: int) -> pd.DataFrame:
    """Run a simple backtest using predicted returns.

    Each day, rank all available stocks by predicted return in descending
    order and take an equal‑weighted long position in the top ``topk``
    stocks.  The strategy return for day t is the average of the
    realised next‑day returns of the selected stocks.

    Parameters
    ----------
    preds : pd.Series
        Predicted returns indexed by (date, code).
    returns : pd.Series
        Realised next‑day returns indexed by (date, code).
    topk : int
        Number of stocks to hold each day.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['strategy_return', 'cumulative_return',
        'daily_mean', 'daily_std', 'sharpe', 'max_drawdown'] indexed by date.
    """
    # Align predictions and returns
    idx = preds.index.intersection(returns.index)
    preds = preds.loc[idx]
    returns = returns.loc[idx]
    # Group by date
    daily_returns: Dict[dt.date, float] = {}
    for d, group in preds.groupby(level=0):
        # sort by prediction descending
        group_sorted = group.sort_values(ascending=False)
        selected_codes = group_sorted.index.get_level_values(1)[:topk]
        # compute realised return of selected codes
        rts = returns.loc[(d, selected_codes)]
        if len(rts) == 0:
            daily_returns[d] = 0.0
        else:
            daily_returns[d] = rts.mean()
    # Build DataFrame
    sr = pd.Series(daily_returns).sort_index()
    cum_ret = (1 + sr).cumprod() - 1
    # Performance metrics
    daily_mean = sr.mean()
    daily_std = sr.std(ddof=1)
    sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std != 0 else np.nan
    # Compute drawdown
    cumulative = (1 + sr).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    # Build result DataFrame
    results = pd.DataFrame({
        'strategy_return': sr,
        'cumulative_return': cum_ret,
    })
    results['daily_mean'] = daily_mean
    results['daily_std'] = daily_std
    results['sharpe'] = sharpe
    results['max_drawdown'] = max_drawdown
    return results


def main() -> None:
    args = parse_args()
    # Load data
    feats, rets = load_data(args.features_path, args.returns_path)
    # Filter date ranges
    idx_dates = feats.index.get_level_values(0)
    train_dates = (args.train_start, args.train_end)
    valid_dates = (args.valid_start, args.valid_end)
    test_dates = (args.test_start, args.test_end)
    # Train model and obtain test predictions
    preds, meta = train_model(feats, rets, algorithm=args.algorithm,
                              train_dates=train_dates, valid_dates=valid_dates,
                              test_dates=test_dates)
    # Backtest
    results = backtest(preds, rets.loc[preds.index], args.topk)
    # Output summary
    print("Backtest summary:\n")
    last_day = results.index[-1]
    total_return = results.loc[last_day, 'cumulative_return']
    print(f"Total return: {total_return:.2%}")
    print(f"Annualised return: {(1 + total_return) ** (252 / len(results)) - 1:.2%}")
    print(f"Sharpe ratio: {results['sharpe'].iloc[0]:.2f}")
    print(f"Max drawdown: {results['max_drawdown'].iloc[0]:.2%}")
    # Save daily results
    out_path = 'backtest_results.csv'
    results.to_csv(out_path)
    print(f"Daily backtest results saved to {out_path}")


if __name__ == "__main__":
    main()