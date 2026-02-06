#!/usr/bin/env python3
"""
CLI for training the regime classifier.

Usage:
    python -m ml.training.regime_trainer --symbol SOL/USDT --days 90
    python -m ml.training.regime_trainer --symbol ETH/USDT --days 60 --save
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import yaml

from ml.models.regime_classifier import RegimeClassifier
from ml.features.regime_features import extract_regime_features
from ml.training.label_generator import generate_regime_labels, create_training_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_training_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data for training using cached backtest data fetcher."""
    from backtest.data_fetcher import fetch_ohlcv
    
    logger.info(f"Fetching {days} days of data for {symbol}...")
    
    # Use the backtest data fetcher - it has caching and better reliability
    df = fetch_ohlcv(symbol, timeframe='1h', days=days)
    
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")
    
    # Reset index to get timestamp as column (data_fetcher returns datetime index)
    df = df.reset_index()
    df = df.rename(columns={'index': 'timestamp'})
    
    logger.info(f"Fetched {len(df)} candles")
    return df


def main():
    parser = argparse.ArgumentParser(description='Train regime classifier')
    parser.add_argument('--symbol', type=str, default='SOL/USDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=30, help='Days of data to fetch')
    parser.add_argument('--forward-hours', type=int, default=24, help='Forward-looking hours for labels')
    parser.add_argument('--save', action='store_true', help='Save the trained model')
    parser.add_argument('--output', type=str, default='data/models/regime_classifier.joblib',
                        help='Output path for model')

    args = parser.parse_args()

    # Load ML config
    ml_config_path = 'config/ml_config.yaml'
    ml_config = {}
    if os.path.exists(ml_config_path):
        with open(ml_config_path, 'r') as f:
            ml_config = yaml.safe_load(f)

    # Fetch data
    try:
        df = fetch_training_data(args.symbol, args.days)
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)

    # Create training dataset
    logger.info("Generating features and labels...")

    try:
        features_df, labels = create_training_dataset(
            df,
            feature_extractor=extract_regime_features,
            label_generator=lambda d, p: generate_regime_labels(d, p),
            window=100,
            forward_period=args.forward_hours,
        )
    except Exception as e:
        logger.error(f"Failed to create training dataset: {e}")
        sys.exit(1)

    logger.info(f"Training dataset: {len(features_df)} samples")
    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")

    if len(features_df) < 100:
        logger.error("Insufficient training data (need at least 100 samples)")
        sys.exit(1)

    # Train classifier
    classifier = RegimeClassifier(ml_config.get('regime_classifier', {}))

    try:
        metrics = classifier.train(features_df, labels)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")

    print("\nFeature Importance:")
    importance = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in importance[:10]:
        print(f"  {feat}: {imp:.4f}")

    print("\nClassification Report:")
    report = metrics['classification_report']
    for label in ['0', '1']:
        if label in report:
            r = report[label]
            print(f"  Class {label}: precision={r['precision']:.3f}, recall={r['recall']:.3f}, f1={r['f1-score']:.3f}")

    # Save model
    if args.save:
        logger.info(f"Saving model to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if classifier.save(args.output):
            print(f"\nModel saved to: {args.output}")
        else:
            logger.error("Failed to save model")
            sys.exit(1)


if __name__ == '__main__':
    main()
