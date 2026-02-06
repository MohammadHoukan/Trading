#!/usr/bin/env python3
"""
CLI for training the pair profitability predictor.

Usage:
    python -m ml.training.pair_trainer --days 90
    python -m ml.training.pair_trainer --days 60 --save
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import yaml

from shared.config import load_config
from shared.database import Database
from ml.models.pair_predictor import PairPredictor
from ml.features.pair_features import extract_pair_features
from ml.training.label_generator import generate_pair_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default pairs for training
DEFAULT_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
    'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT',
    'DOT/USDT', 'POL/USDT', 'LINK/USDT', 'LTC/USDT',
]


def fetch_market_data(symbols: list, config: dict) -> pd.DataFrame:
    """Fetch current market data for all symbols."""
    from workers.order_manager import OrderManager

    exchange_cfg = config.get('exchange', {})
    om = OrderManager(
        exchange_cfg.get('name', 'binance'),
        exchange_cfg.get('api_key', ''),
        exchange_cfg.get('secret', ''),
        testnet=(exchange_cfg.get('mode') == 'testnet')
    )

    data = []
    for symbol in symbols:
        try:
            ticker = om.fetch_ticker(symbol)
            if ticker:
                volume_24h = ticker.get('quoteVolume', 0) or 0
                bid = ticker.get('bid', 0) or 0
                ask = ticker.get('ask', 0) or 0
                mid = (bid + ask) / 2 if bid and ask else 0
                spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999

                # Get fees
                maker_fee = 0.1
                taker_fee = 0.1
                try:
                    markets = om.exchange.load_markets()
                    if symbol in markets:
                        market = markets[symbol]
                        maker_fee = market.get('maker', 0.001) * 100
                        taker_fee = market.get('taker', 0.001) * 100
                except Exception:
                    pass

                data.append({
                    'symbol': symbol,
                    'volume_24h': volume_24h,
                    'spread_pct': spread_pct,
                    'maker_fee': maker_fee,
                    'taker_fee': taker_fee,
                })
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")

    return pd.DataFrame(data)


def fetch_historical_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data for a symbol using cached backtest data fetcher."""
    from backtest.data_fetcher import fetch_ohlcv
    
    try:
        df = fetch_ohlcv(symbol, timeframe='1h', days=days)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Reset index to get timestamp as column
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch historical data for {symbol}: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Train pair profitability predictor')
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_PAIRS, help='Symbols to train on')
    parser.add_argument('--days', type=int, default=30, help='Days of data for metrics')
    parser.add_argument('--forward-days', type=int, default=7, help='Forward-looking days for labels')
    parser.add_argument('--save', action='store_true', help='Save the trained model')
    parser.add_argument('--output', type=str, default='data/models/pair_predictor.joblib',
                        help='Output path for model')
    parser.add_argument('--config', type=str, default='config/settings.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config = {}

    # Load ML config
    ml_config_path = 'config/ml_config.yaml'
    ml_config = {}
    if os.path.exists(ml_config_path):
        with open(ml_config_path, 'r') as f:
            ml_config = yaml.safe_load(f)

    # Initialize database
    db = Database()

    # Fetch market data
    logger.info(f"Fetching market data for {len(args.symbols)} symbols...")
    market_df = fetch_market_data(args.symbols, config)
    logger.info(f"Got market data for {len(market_df)} symbols")

    if len(market_df) == 0:
        logger.error("No market data available")
        sys.exit(1)

    # Build training dataset
    features_list = []
    labels_list = []

    for symbol in args.symbols:
        logger.info(f"Processing {symbol}...")

        # Get market data
        market_row = market_df[market_df['symbol'] == symbol]
        if market_row.empty:
            continue

        market_data = market_row.iloc[0].to_dict()

        # Get backtest metrics from database
        backtest_metrics = db.get_backtest_metrics(symbol, args.days)

        # Fetch historical data for label generation
        try:
            ohlcv = fetch_historical_data(symbol, args.days + args.forward_days + 10)
            if len(ohlcv) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Generate profitability labels
            labels = generate_pair_labels(ohlcv, forward_days=args.forward_days)

            # For training, we use multiple time points
            for i in range(0, len(ohlcv) - args.forward_days * 24, 24):  # Sample every 24 hours
                if pd.isna(labels.iloc[i]):
                    continue

                # Extract features
                features = extract_pair_features(market_data, backtest_metrics)
                features_list.append(features)
                labels_list.append(labels.iloc[i])

        except Exception as e:
            logger.warning(f"Failed to process {symbol}: {e}")
            continue

    if not features_list:
        logger.error("No training data generated")
        sys.exit(1)

    features_df = pd.DataFrame(features_list)
    labels_series = pd.Series(labels_list)

    # Remove samples with missing data
    mask = ~(features_df.isna().any(axis=1) | labels_series.isna())
    features_df = features_df[mask]
    labels_series = labels_series[mask]

    logger.info(f"Training dataset: {len(features_df)} samples")
    logger.info(f"Label range: [{labels_series.min():.4f}, {labels_series.max():.4f}]")

    if len(features_df) < 50:
        logger.error("Insufficient training data (need at least 50 samples)")
        sys.exit(1)

    # Train predictor
    predictor = PairPredictor(ml_config.get('pair_predictor', {}))

    try:
        metrics = predictor.train(features_df, labels_series)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"R² Score: {metrics['r2']:.3f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
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

    # Save model
    if args.save:
        logger.info(f"Saving model to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if predictor.save(args.output):
            print(f"\nModel saved to: {args.output}")
        else:
            logger.error("Failed to save model")
            sys.exit(1)


if __name__ == '__main__':
    main()
