"""
Label generation for ML training data.

Generates forward-looking labels by simulating trading outcomes
or measuring future price movements.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_regime_labels(
    df: pd.DataFrame,
    forward_hours: int = 24,
    grid_spacing_pct: float = 0.02,
    min_profit_threshold: float = 0.001,
) -> pd.Series:
    """
    Generate regime labels based on simulated grid trading profitability.

    For each time point, simulate a simple grid strategy forward and
    label as FAVORABLE (1) if profitable, UNFAVORABLE (0) otherwise.

    Args:
        df: OHLCV DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        forward_hours: How many hours to look forward for outcome
        grid_spacing_pct: Grid spacing as percentage of price
        min_profit_threshold: Minimum profit % to label as FAVORABLE

    Returns:
        Series of labels (1 = FAVORABLE, 0 = UNFAVORABLE)
    """
    labels = []

    for i in range(len(df) - forward_hours):
        try:
            entry_price = df.iloc[i]['close']
            future_data = df.iloc[i + 1:i + forward_hours + 1]

            if len(future_data) < forward_hours:
                labels.append(np.nan)
                continue

            # Simulate simple grid outcome
            profit = _simulate_grid_profit(
                entry_price,
                future_data,
                grid_spacing_pct
            )

            # Label based on profit threshold
            if profit >= min_profit_threshold:
                labels.append(1)  # FAVORABLE
            else:
                labels.append(0)  # UNFAVORABLE

        except Exception as e:
            logger.warning(f"Label generation error at index {i}: {e}")
            labels.append(np.nan)

    # Pad with NaN for forward-looking indices
    labels.extend([np.nan] * forward_hours)

    return pd.Series(labels, index=df.index)


def _simulate_grid_profit(
    entry_price: float,
    future_data: pd.DataFrame,
    grid_spacing_pct: float,
) -> float:
    """
    Simulate simplified grid trading to estimate profit.

    This is a simplified model:
    - Count how many times price crosses grid levels
    - Each crossing earns grid_spacing_pct profit

    Args:
        entry_price: Starting price
        future_data: Future OHLCV data
        grid_spacing_pct: Grid spacing as percentage

    Returns:
        Estimated profit as percentage of capital
    """
    grid_spacing = entry_price * grid_spacing_pct
    profit = 0.0

    # Track crossings
    last_level = int(entry_price / grid_spacing)

    for _, row in future_data.iterrows():
        high = row['high']
        low = row['low']

        high_level = int(high / grid_spacing)
        low_level = int(low / grid_spacing)

        # Count level crossings (each crossing = potential profit)
        crossings = abs(high_level - low_level)
        if crossings > 0:
            # Each crossing earns approximately half the grid spacing
            # (buy low, sell high within the range)
            profit += crossings * grid_spacing_pct * 0.5

    return profit


def generate_pair_labels(
    df: pd.DataFrame,
    forward_days: int = 7,
    capital: float = 1000.0,
    grid_levels: int = 20,
) -> pd.Series:
    """
    Generate pair profitability labels for training.

    For each data point, calculate the return that would have been
    achieved by trading the pair over the forward period.

    Args:
        df: OHLCV DataFrame
        forward_days: Days to look forward
        capital: Notional capital for calculation
        grid_levels: Number of grid levels to simulate

    Returns:
        Series of profitability scores (continuous)
    """
    forward_hours = forward_days * 24
    labels = []

    for i in range(len(df) - forward_hours):
        try:
            future_data = df.iloc[i:i + forward_hours + 1]

            if len(future_data) < forward_hours:
                labels.append(np.nan)
                continue

            # Calculate metrics
            entry_price = future_data.iloc[0]['close']
            exit_price = future_data.iloc[-1]['close']

            # Price range (volatility)
            price_range = (future_data['high'].max() - future_data['low'].min()) / entry_price

            # Number of crosses through the range center
            mid_price = (future_data['high'].max() + future_data['low'].min()) / 2
            crosses = sum(
                1 for j in range(1, len(future_data))
                if (future_data.iloc[j - 1]['close'] < mid_price and future_data.iloc[j]['close'] >= mid_price) or
                   (future_data.iloc[j - 1]['close'] >= mid_price and future_data.iloc[j]['close'] < mid_price)
            )

            # Profitability score combines range trading potential with volatility
            # More crosses = more trades = more potential profit
            # Higher range = larger profit per trade
            grid_profit_estimate = (price_range / grid_levels) * crosses * 0.5

            # Adjust for ending position (unrealized P&L)
            position_pnl = (exit_price - entry_price) / entry_price * 0.1  # Small weight

            profit_score = grid_profit_estimate + position_pnl
            labels.append(profit_score)

        except Exception as e:
            logger.warning(f"Pair label generation error at index {i}: {e}")
            labels.append(np.nan)

    # Pad with NaN
    labels.extend([np.nan] * forward_hours)

    return pd.Series(labels, index=df.index)


def create_training_dataset(
    df: pd.DataFrame,
    feature_extractor,
    label_generator,
    window: int = 100,
    forward_period: int = 24,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a complete training dataset with features and labels.

    Args:
        df: Full OHLCV DataFrame
        feature_extractor: Function to extract features from window
        label_generator: Function to generate labels
        window: Feature extraction window size
        forward_period: Forward-looking period for labels

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    # Generate labels for all valid points
    labels = label_generator(df, forward_period)

    # Extract features for matching points
    features_list = []
    valid_indices = []

    for i in range(window, len(df) - forward_period):
        try:
            window_df = df.iloc[i - window:i + 1].reset_index(drop=True)
            features = feature_extractor(window_df)
            features_list.append(features)
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Feature extraction error at {i}: {e}")
            continue

    features_df = pd.DataFrame(features_list)
    features_df.index = valid_indices

    # Align labels with features
    aligned_labels = labels.loc[valid_indices]

    # Drop rows with missing values
    mask = ~(features_df.isna().any(axis=1) | aligned_labels.isna())
    features_df = features_df[mask]
    aligned_labels = aligned_labels[mask]

    return features_df, aligned_labels
