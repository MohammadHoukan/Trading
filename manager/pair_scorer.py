"""
Pair Scoring System for systematic trading pair selection.

Evaluates pairs based on:
- Liquidity (24h volume)
- Spread (bid-ask spread)
- Fees (maker/taker fees)
- Historical fill rate (from grid_events database)

Usage:
    from manager.pair_scorer import PairScorer
    scorer = PairScorer(config)
    scores = scorer.score_pairs(['SOL/USDT', 'ETH/USDT', 'BTC/USDT'])
    print(scores)
"""

import logging
import sqlite3
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from workers.order_manager import OrderManager


@dataclass
class PairScore:
    """Score breakdown for a trading pair."""
    symbol: str
    volume_24h: float
    spread_pct: float
    maker_fee_pct: float
    taker_fee_pct: float
    fill_rate: Optional[float]  # None if no historical data
    sample_size: int

    # Individual scores (0-100)
    volume_score: float
    spread_score: float
    fee_score: float
    fill_rate_score: float

    # Composite score
    composite_score: float

    # Recommendation
    recommendation: str  # 'STRONG', 'GOOD', 'MARGINAL', 'AVOID'


class PairScorer:
    """
    Systematic pair evaluation for grid trading.

    Higher scores = better for grid trading.
    """

    def __init__(self, config: dict, db_path: str = 'swarm.db'):
        self.logger = logging.getLogger("PairScorer")
        self.config = config
        self.db_path = db_path

        # Initialize exchange connection for market data
        self.exchange = OrderManager(
            config['exchange']['name'],
            config['exchange']['api_key'],
            config['exchange']['secret'],
            testnet=(config['exchange']['mode'] == 'testnet')
        )

        # Scoring thresholds (configurable)
        scorer_cfg = config.get('pair_scorer', {})
        self.thresholds = {
            'min_volume_24h': scorer_cfg.get('min_volume_24h', 1_000_000),  # $1M
            'ideal_volume_24h': scorer_cfg.get('ideal_volume_24h', 10_000_000),  # $10M
            'max_spread_pct': scorer_cfg.get('max_spread_pct', 0.5),  # 0.5%
            'ideal_spread_pct': scorer_cfg.get('ideal_spread_pct', 0.1),  # 0.1%
            'max_fee_pct': scorer_cfg.get('max_fee_pct', 0.2),  # 0.2%
            'ideal_fee_pct': scorer_cfg.get('ideal_fee_pct', 0.05),  # 0.05%
            'min_fill_rate': scorer_cfg.get('min_fill_rate', 0.2),  # 20%
            'ideal_fill_rate': scorer_cfg.get('ideal_fill_rate', 0.5),  # 50%
        }

        # Weights for composite score
        self.weights = {
            'volume': scorer_cfg.get('weight_volume', 0.25),
            'spread': scorer_cfg.get('weight_spread', 0.30),
            'fees': scorer_cfg.get('weight_fees', 0.20),
            'fill_rate': scorer_cfg.get('weight_fill_rate', 0.25),
        }

    def score_pairs(self, symbols: List[str]) -> List[PairScore]:
        """
        Score multiple trading pairs.

        Args:
            symbols: List of trading pairs (e.g., ['SOL/USDT', 'ETH/USDT'])

        Returns:
            List of PairScore objects, sorted by composite score (descending)
        """
        scores = []
        for symbol in symbols:
            try:
                score = self._score_pair(symbol)
                if score:
                    scores.append(score)
            except Exception as e:
                self.logger.error(f"Failed to score {symbol}: {e}")

        # Sort by composite score (highest first)
        scores.sort(key=lambda x: x.composite_score, reverse=True)
        return scores

    def _score_pair(self, symbol: str) -> Optional[PairScore]:
        """Score a single trading pair."""

        # 1. Fetch market data
        market_data = self._fetch_market_data(symbol)
        if not market_data:
            self.logger.warning(f"Could not fetch market data for {symbol}")
            return None

        volume_24h = market_data.get('volume_24h', 0)
        spread_pct = market_data.get('spread_pct', 999)
        maker_fee = market_data.get('maker_fee', 0.1)
        taker_fee = market_data.get('taker_fee', 0.1)

        # 2. Fetch historical fill rate from database
        fill_data = self._fetch_fill_rate(symbol)
        fill_rate = fill_data.get('fill_rate')
        sample_size = fill_data.get('sample_size', 0)

        # 3. Calculate individual scores (0-100)
        volume_score = self._score_volume(volume_24h)
        spread_score = self._score_spread(spread_pct)
        fee_score = self._score_fees((maker_fee + taker_fee) / 2)
        fill_rate_score = self._score_fill_rate(fill_rate)

        # 4. Calculate composite score
        # If no fill rate data, use only volume/spread/fees
        if fill_rate is None:
            total_weight = self.weights['volume'] + self.weights['spread'] + self.weights['fees']
            composite = (
                volume_score * self.weights['volume'] +
                spread_score * self.weights['spread'] +
                fee_score * self.weights['fees']
            ) / total_weight
        else:
            composite = (
                volume_score * self.weights['volume'] +
                spread_score * self.weights['spread'] +
                fee_score * self.weights['fees'] +
                fill_rate_score * self.weights['fill_rate']
            )

        # 5. Determine recommendation
        recommendation = self._get_recommendation(composite, volume_24h, spread_pct)

        return PairScore(
            symbol=symbol,
            volume_24h=volume_24h,
            spread_pct=spread_pct,
            maker_fee_pct=maker_fee,
            taker_fee_pct=taker_fee,
            fill_rate=fill_rate,
            sample_size=sample_size,
            volume_score=round(volume_score, 1),
            spread_score=round(spread_score, 1),
            fee_score=round(fee_score, 1),
            fill_rate_score=round(fill_rate_score, 1),
            composite_score=round(composite, 1),
            recommendation=recommendation,
        )

    def _fetch_market_data(self, symbol: str) -> Dict:
        """Fetch volume, spread, and fees from exchange."""
        try:
            # Fetch ticker for volume and spread
            ticker = self.exchange.fetch_ticker(symbol)
            if not ticker:
                return {}

            # Calculate volume in USD (quote currency)
            volume_24h = ticker.get('quoteVolume', 0) or 0

            # Calculate spread
            bid = ticker.get('bid', 0) or 0
            ask = ticker.get('ask', 0) or 0
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999

            # Get fees (most exchanges have this in markets)
            # Default to typical Binance fees if not available
            maker_fee = 0.1  # 0.1% default
            taker_fee = 0.1

            try:
                markets = self.exchange.exchange.load_markets()
                if symbol in markets:
                    market = markets[symbol]
                    maker_fee = market.get('maker', 0.001) * 100
                    taker_fee = market.get('taker', 0.001) * 100
            except Exception:
                pass

            return {
                'volume_24h': volume_24h,
                'spread_pct': spread_pct,
                'maker_fee': maker_fee,
                'taker_fee': taker_fee,
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return {}

    def _fetch_fill_rate(self, symbol: str, lookback_hours: int = 168) -> Dict:
        """
        Fetch historical fill rate from grid_events database.

        Args:
            symbol: Trading pair
            lookback_hours: How far back to look (default 7 days)

        Returns:
            Dict with 'fill_rate' (float or None) and 'sample_size' (int)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            since = time.time() - (lookback_hours * 3600)
            cursor.execute('''
                SELECT
                    SUM(CASE WHEN event_type = 'FILL' THEN 1 ELSE 0 END) as fills,
                    SUM(CASE WHEN event_type = 'PLACE' THEN 1 ELSE 0 END) as places
                FROM grid_events
                WHERE symbol = ? AND timestamp > ? AND source = 'live'
            ''', (symbol, since))

            row = cursor.fetchone()
            conn.close()

            if row and row[1] and row[1] > 0:
                return {
                    'fill_rate': row[0] / row[1],
                    'sample_size': row[1],
                }
            return {'fill_rate': None, 'sample_size': 0}

        except Exception as e:
            self.logger.warning(f"Failed to fetch fill rate for {symbol}: {e}")
            return {'fill_rate': None, 'sample_size': 0}

    def _score_volume(self, volume_24h: float) -> float:
        """Score based on 24h volume. Higher volume = better."""
        if volume_24h <= 0:
            return 0
        if volume_24h >= self.thresholds['ideal_volume_24h']:
            return 100
        if volume_24h <= self.thresholds['min_volume_24h']:
            return max(0, (volume_24h / self.thresholds['min_volume_24h']) * 30)

        # Linear interpolation between min and ideal
        ratio = (volume_24h - self.thresholds['min_volume_24h']) / (
            self.thresholds['ideal_volume_24h'] - self.thresholds['min_volume_24h']
        )
        return 30 + (ratio * 70)

    def _score_spread(self, spread_pct: float) -> float:
        """Score based on spread. Lower spread = better."""
        if spread_pct <= self.thresholds['ideal_spread_pct']:
            return 100
        if spread_pct >= self.thresholds['max_spread_pct']:
            return 0

        # Linear interpolation (inverted)
        ratio = (self.thresholds['max_spread_pct'] - spread_pct) / (
            self.thresholds['max_spread_pct'] - self.thresholds['ideal_spread_pct']
        )
        return ratio * 100

    def _score_fees(self, avg_fee_pct: float) -> float:
        """Score based on fees. Lower fees = better."""
        if avg_fee_pct <= self.thresholds['ideal_fee_pct']:
            return 100
        if avg_fee_pct >= self.thresholds['max_fee_pct']:
            return 0

        # Linear interpolation (inverted)
        ratio = (self.thresholds['max_fee_pct'] - avg_fee_pct) / (
            self.thresholds['max_fee_pct'] - self.thresholds['ideal_fee_pct']
        )
        return ratio * 100

    def _score_fill_rate(self, fill_rate: Optional[float]) -> float:
        """Score based on historical fill rate. Higher = better."""
        if fill_rate is None:
            return 50  # Neutral score if no data
        if fill_rate >= self.thresholds['ideal_fill_rate']:
            return 100
        if fill_rate <= self.thresholds['min_fill_rate']:
            return max(0, (fill_rate / self.thresholds['min_fill_rate']) * 30)

        # Linear interpolation
        ratio = (fill_rate - self.thresholds['min_fill_rate']) / (
            self.thresholds['ideal_fill_rate'] - self.thresholds['min_fill_rate']
        )
        return 30 + (ratio * 70)

    def _get_recommendation(self, score: float, volume: float, spread: float) -> str:
        """Determine trading recommendation based on score and hard limits."""
        # Hard limits that override score
        if volume < self.thresholds['min_volume_24h'] * 0.5:
            return 'AVOID'  # Too illiquid
        if spread > self.thresholds['max_spread_pct']:
            return 'AVOID'  # Spread too wide

        # Score-based recommendation
        if score >= 75:
            return 'STRONG'
        elif score >= 55:
            return 'GOOD'
        elif score >= 40:
            return 'MARGINAL'
        else:
            return 'AVOID'

    def print_report(self, scores: List[PairScore]):
        """Print a formatted report of pair scores."""
        print("\n" + "=" * 80)
        print("PAIR SCORING REPORT")
        print("=" * 80)
        print(f"{'Symbol':<12} {'Score':>6} {'Rec':>10} {'Volume':>12} {'Spread':>8} {'Fill%':>8} {'Samples':>8}")
        print("-" * 80)

        for s in scores:
            fill_str = f"{s.fill_rate*100:.1f}%" if s.fill_rate else "N/A"
            vol_str = f"${s.volume_24h/1_000_000:.1f}M"
            print(f"{s.symbol:<12} {s.composite_score:>6.1f} {s.recommendation:>10} {vol_str:>12} {s.spread_pct:>7.3f}% {fill_str:>8} {s.sample_size:>8}")

        print("=" * 80)
        print("\nScore Breakdown (weights):")
        print(f"  Volume: {self.weights['volume']*100:.0f}% | Spread: {self.weights['spread']*100:.0f}% | Fees: {self.weights['fees']*100:.0f}% | Fill Rate: {self.weights['fill_rate']*100:.0f}%")
        print("\nRecommendations: STRONG (>=75) | GOOD (>=55) | MARGINAL (>=40) | AVOID (<40)")


# CLI entry point
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from shared.config import load_config

    # Default pairs to analyze
    DEFAULT_PAIRS = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
        'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT',
        'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'LTC/USDT',
    ]

    config = load_config()
    scorer = PairScorer(config)

    # Get pairs from command line or use defaults
    pairs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PAIRS

    print(f"Analyzing {len(pairs)} pairs...")
    scores = scorer.score_pairs(pairs)
    scorer.print_report(scores)
