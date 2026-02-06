"""
Feature engineering for regime classification.

Extracts technical indicators and market structure features
for training the regime classifier.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def extract_regime_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features for regime classification from OHLCV data.

    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Should have at least 100 rows for reliable indicator calculation.

    Returns:
        Dict of feature name -> value
    """
    if len(df) < 50:
        raise ValueError("Need at least 50 candles for feature extraction")

    features = {}

    # ===== ADX (Trend Strength) =====
    adx_df = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)
    adx_col = next((c for c in adx_df.columns if str(c).startswith('ADX_')), None)
    features['adx'] = adx_df.iloc[-1][adx_col] if adx_col else np.nan

    # ===== ATR/Price (Volatility) =====
    atr = df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
    current_price = df['close'].iloc[-1]
    features['atr_pct'] = (atr.iloc[-1] / current_price) if atr is not None and not atr.empty else np.nan

    # ===== MA Distance =====
    sma_20 = df['close'].rolling(20).mean().iloc[-1]
    sma_50 = df['close'].rolling(50).mean().iloc[-1]
    features['ma_distance_20'] = (current_price - sma_20) / sma_20 if sma_20 else np.nan
    features['ma_distance_50'] = (current_price - sma_50) / sma_50 if sma_50 else np.nan

    # ===== RSI =====
    rsi = df.ta.rsi(close=df['close'], length=14)
    features['rsi'] = rsi.iloc[-1] if rsi is not None and not rsi.empty else np.nan

    # ===== MACD =====
    macd = df.ta.macd(close=df['close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        hist_col = next((c for c in macd.columns if 'HIST' in str(c) or 'MACDh' in str(c)), None)
        if hist_col:
            features['macd_histogram'] = macd.iloc[-1][hist_col]
            # Normalize by price
            features['macd_histogram_pct'] = features['macd_histogram'] / current_price
        else:
            features['macd_histogram'] = np.nan
            features['macd_histogram_pct'] = np.nan
    else:
        features['macd_histogram'] = np.nan
        features['macd_histogram_pct'] = np.nan

    # ===== Volume Ratio =====
    avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    features['volume_ratio'] = current_volume / avg_volume_20 if avg_volume_20 > 0 else np.nan

    # ===== Bollinger Band Width =====
    bbands = df.ta.bbands(close=df['close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        upper_col = next((c for c in bbands.columns if 'BBU' in str(c)), None)
        lower_col = next((c for c in bbands.columns if 'BBL' in str(c)), None)
        if upper_col and lower_col:
            bb_upper = bbands.iloc[-1][upper_col]
            bb_lower = bbands.iloc[-1][lower_col]
            features['bb_width'] = (bb_upper - bb_lower) / current_price
        else:
            features['bb_width'] = np.nan
    else:
        features['bb_width'] = np.nan

    # ===== Price Position in Range =====
    high_20 = df['high'].rolling(20).max().iloc[-1]
    low_20 = df['low'].rolling(20).min().iloc[-1]
    range_20 = high_20 - low_20
    features['price_position'] = (current_price - low_20) / range_20 if range_20 > 0 else 0.5

    # ===== Time Features =====
    if 'timestamp' in df.columns:
        last_ts = df['timestamp'].iloc[-1]
        if isinstance(last_ts, (int, float)):
            dt = datetime.fromtimestamp(last_ts / 1000)  # Assuming milliseconds
        else:
            dt = pd.to_datetime(last_ts)
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()
    else:
        features['hour_of_day'] = np.nan
        features['day_of_week'] = np.nan

    # ===== Recent Returns =====
    features['return_1h'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) if len(df) > 1 else np.nan
    features['return_4h'] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) if len(df) > 4 else np.nan
    features['return_24h'] = (df['close'].iloc[-1] / df['close'].iloc[-25] - 1) if len(df) > 24 else np.nan

    return features


def extract_regime_features_batch(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Extract features for multiple time points (rolling window).

    Args:
        df: Full OHLCV DataFrame
        window: Window size for feature extraction

    Returns:
        DataFrame with features for each valid time point
    """
    features_list = []
    timestamps = []

    for i in range(window, len(df)):
        window_df = df.iloc[i - window:i + 1].reset_index(drop=True)
        try:
            features = extract_regime_features(window_df)
            features_list.append(features)
            timestamps.append(df.iloc[i]['timestamp'])
        except Exception:
            continue

    result = pd.DataFrame(features_list)
    result['timestamp'] = timestamps
    return result


def get_feature_names() -> List[str]:
    """Return list of feature names in order."""
    return [
        'adx',
        'atr_pct',
        'ma_distance_20',
        'ma_distance_50',
        'rsi',
        'macd_histogram_pct',
        'volume_ratio',
        'bb_width',
        'price_position',
        'hour_of_day',
        'day_of_week',
        'return_1h',
        'return_4h',
        'return_24h',
    ]
