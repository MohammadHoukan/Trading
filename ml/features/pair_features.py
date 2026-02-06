"""
Feature engineering for pair profitability prediction.

Extracts market characteristics and historical performance metrics
for predicting pair trading profitability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def extract_pair_features(
    market_data: Dict,
    backtest_metrics: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Extract features for pair profitability prediction.

    Args:
        market_data: Dict with market characteristics:
            - volume_24h: 24-hour trading volume
            - spread_pct: Bid-ask spread percentage
            - maker_fee: Maker fee percentage
            - taker_fee: Taker fee percentage
        backtest_metrics: Dict with historical performance (optional):
            - sharpe_ratio: Risk-adjusted return
            - win_rate: Percentage of profitable trades
            - max_drawdown: Maximum drawdown percentage
            - profit_factor: Gross profit / gross loss
            - avg_trade_return: Average return per trade
            - trade_count: Number of trades in period

    Returns:
        Dict of feature name -> value
    """
    features = {}

    # ===== Market Features =====
    features['volume_24h_log'] = np.log10(max(market_data.get('volume_24h', 1), 1))
    features['spread_pct'] = market_data.get('spread_pct', 0.5)
    features['maker_fee'] = market_data.get('maker_fee', 0.1)
    features['taker_fee'] = market_data.get('taker_fee', 0.1)
    features['avg_fee'] = (features['maker_fee'] + features['taker_fee']) / 2

    # Derived market features
    # Spread-to-fee ratio (higher means spread dominates costs)
    if features['avg_fee'] > 0:
        features['spread_fee_ratio'] = features['spread_pct'] / features['avg_fee']
    else:
        features['spread_fee_ratio'] = features['spread_pct']

    # ===== Backtest Metrics =====
    if backtest_metrics:
        features['sharpe_ratio'] = backtest_metrics.get('sharpe_ratio', 0.0)
        features['win_rate'] = backtest_metrics.get('win_rate', 0.5)
        features['max_drawdown'] = backtest_metrics.get('max_drawdown', 0.0)
        features['profit_factor'] = min(backtest_metrics.get('profit_factor', 1.0), 10.0)  # Cap at 10
        features['avg_trade_return'] = backtest_metrics.get('avg_trade_return', 0.0)
        features['trade_count_log'] = np.log10(max(backtest_metrics.get('trade_count', 1), 1))

        # Derived performance features
        # Consistency score: win_rate * sharpe (rewards consistent profitable strategies)
        features['consistency_score'] = features['win_rate'] * max(features['sharpe_ratio'], 0)

        # Risk-adjusted profit: profit_factor adjusted for drawdown
        if features['max_drawdown'] > 0:
            features['risk_adj_profit'] = features['profit_factor'] / (1 + features['max_drawdown'])
        else:
            features['risk_adj_profit'] = features['profit_factor']
    else:
        # Fill with neutral values if no backtest data
        features['sharpe_ratio'] = 0.0
        features['win_rate'] = 0.5
        features['max_drawdown'] = 0.0
        features['profit_factor'] = 1.0
        features['avg_trade_return'] = 0.0
        features['trade_count_log'] = 0.0
        features['consistency_score'] = 0.0
        features['risk_adj_profit'] = 1.0

    return features


def extract_pair_features_from_df(
    ticker_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Extract features for multiple pairs from DataFrames.

    Args:
        ticker_df: DataFrame with columns ['symbol', 'volume_24h', 'spread_pct', 'maker_fee', 'taker_fee']
        metrics_df: Optional DataFrame with columns ['symbol', 'sharpe_ratio', 'win_rate', ...]

    Returns:
        DataFrame with features for each pair
    """
    features_list = []

    for _, row in ticker_df.iterrows():
        symbol = row['symbol']
        market_data = {
            'volume_24h': row.get('volume_24h', 0),
            'spread_pct': row.get('spread_pct', 0.5),
            'maker_fee': row.get('maker_fee', 0.1),
            'taker_fee': row.get('taker_fee', 0.1),
        }

        backtest_metrics = None
        if metrics_df is not None and symbol in metrics_df['symbol'].values:
            metrics_row = metrics_df[metrics_df['symbol'] == symbol].iloc[0]
            backtest_metrics = metrics_row.to_dict()

        features = extract_pair_features(market_data, backtest_metrics)
        features['symbol'] = symbol
        features_list.append(features)

    return pd.DataFrame(features_list)


def get_feature_names() -> List[str]:
    """Return list of feature names in order."""
    return [
        'volume_24h_log',
        'spread_pct',
        'maker_fee',
        'taker_fee',
        'avg_fee',
        'spread_fee_ratio',
        'sharpe_ratio',
        'win_rate',
        'max_drawdown',
        'profit_factor',
        'avg_trade_return',
        'trade_count_log',
        'consistency_score',
        'risk_adj_profit',
    ]
