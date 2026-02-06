"""
Pair Scoring System for systematic trading pair selection.

Evaluates pairs based on:
- Liquidity (24h volume)
- Spread (bid-ask spread)
- Fees (maker/taker fees)
- Historical fill rate (from grid_events database)
- Backtest metrics (sharpe, win rate, drawdown, profit factor)
- ML profitability prediction (optional)

Usage:
    from manager.pair_scorer import PairScorer
    scorer = PairScorer(config)
    scores = scorer.score_pairs(['SOL/USDT', 'ETH/USDT', 'BTC/USDT'])
    print(scores)
"""

import logging
import sqlite3
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from workers.order_manager import OrderManager
from shared.database import Database

# ML imports (optional)
try:
    from ml.models.pair_predictor import PairPredictor
    from ml.features.pair_features import extract_pair_features
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


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

    # Backtest metrics
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    max_drawdown: float = 0.0
    profit_factor: float = 1.0
    backtest_score: float = 50.0

    # ML prediction
    ml_score: Optional[float] = None

    # Composite score
    composite_score: float = 0.0

    # Recommendation
    recommendation: str = 'MARGINAL'  # 'STRONG', 'GOOD', 'MARGINAL', 'AVOID'


class PairScorer:
    """
    Systematic pair evaluation for grid trading.

    Higher scores = better for grid trading.
    """

    def __init__(self, config: dict, db_path: str = 'swarm.db'):
        self.logger = logging.getLogger("PairScorer")
        self.config = config
        self.db_path = db_path
        self.db = Database(db_path)

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

        # Weights for composite score (updated for multi-factor)
        self.weights = {
            'volume': scorer_cfg.get('weight_volume', 0.15),
            'spread': scorer_cfg.get('weight_spread', 0.20),
            'fees': scorer_cfg.get('weight_fees', 0.10),
            'fill_rate': scorer_cfg.get('weight_fill_rate', 0.15),
            'backtest': scorer_cfg.get('weight_backtest', 0.40),
        }

        # ML configuration
        smart_cfg = config.get('smart_features', {})
        ml_cfg = smart_cfg.get('pair_ranking_ml', {})
        self.ml_enabled = smart_cfg.get('ml_enabled', False) and ml_cfg.get('enabled', False)
        self.ml_weight = ml_cfg.get('ml_weight', 0.4)
        self.ml_model_path = ml_cfg.get('model_path', 'data/models/pair_predictor.joblib')

        # Initialize ML predictor if enabled
        self.ml_predictor = None
        if self.ml_enabled and ML_AVAILABLE:
            self._init_ml_predictor()
        elif self.ml_enabled and not ML_AVAILABLE:
            self.logger.warning("ML features enabled but ml module not available")
            self.ml_enabled = False

    def _init_ml_predictor(self):
        """Initialize and load the ML predictor if available."""
        try:
            self.ml_predictor = PairPredictor(self.config)
            if os.path.exists(self.ml_model_path):
                if self.ml_predictor.load(self.ml_model_path):
                    self.logger.info(f"ML pair predictor loaded from {self.ml_model_path}")
                else:
                    self.logger.warning("Failed to load ML predictor, using rules only")
                    self.ml_enabled = False
            else:
                self.logger.info(f"ML model not found at {self.ml_model_path}, using rules only")
                self.ml_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize ML predictor: {e}")
            self.ml_enabled = False

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

        # 3. Fetch backtest metrics
        backtest_metrics = self._fetch_backtest_metrics(symbol)
        sharpe_ratio = backtest_metrics.get('sharpe_ratio', 0.0)
        win_rate = backtest_metrics.get('win_rate', 0.5)
        max_drawdown = backtest_metrics.get('max_drawdown', 0.0)
        profit_factor = backtest_metrics.get('profit_factor', 1.0)

        # 4. Calculate individual scores (0-100)
        volume_score = self._score_volume(volume_24h)
        spread_score = self._score_spread(spread_pct)
        fee_score = self._score_fees((maker_fee + taker_fee) / 2)
        fill_rate_score = self._score_fill_rate(fill_rate)
        backtest_score = self._score_backtest(backtest_metrics)

        # 5. Calculate rule-based composite score
        weights_sum = (
            self.weights['volume'] +
            self.weights['spread'] +
            self.weights['fees'] +
            (self.weights['fill_rate'] if fill_rate is not None else 0) +
            (self.weights['backtest'] if backtest_metrics.get('trade_count', 0) > 0 else 0)
        )

        rule_composite = (
            volume_score * self.weights['volume'] +
            spread_score * self.weights['spread'] +
            fee_score * self.weights['fees'] +
            (fill_rate_score * self.weights['fill_rate'] if fill_rate is not None else 0) +
            (backtest_score * self.weights['backtest'] if backtest_metrics.get('trade_count', 0) > 0 else 0)
        ) / weights_sum if weights_sum > 0 else 50.0

        # 6. Get ML prediction if enabled
        ml_score = None
        if self.ml_enabled and self.ml_predictor:
            try:
                features = extract_pair_features(market_data, backtest_metrics)
                ml_score = self.ml_predictor.get_normalized_score(features)
            except Exception as e:
                self.logger.warning(f"ML prediction failed for {symbol}: {e}")

        # 7. Ensemble rule-based and ML scores
        if ml_score is not None:
            composite = (1 - self.ml_weight) * rule_composite + self.ml_weight * ml_score
        else:
            composite = rule_composite

        # 8. Determine recommendation
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
            sharpe_ratio=round(sharpe_ratio, 2),
            win_rate=round(win_rate, 3),
            max_drawdown=round(max_drawdown, 4),
            profit_factor=round(profit_factor, 2),
            backtest_score=round(backtest_score, 1),
            ml_score=round(ml_score, 1) if ml_score is not None else None,
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

    def _fetch_backtest_metrics(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Fetch historical backtest metrics from the database.

        Args:
            symbol: Trading pair
            lookback_days: Days to look back for metrics

        Returns:
            Dict with sharpe_ratio, win_rate, max_drawdown, profit_factor, trade_count
        """
        try:
            return self.db.get_backtest_metrics(symbol, lookback_days)
        except Exception as e:
            self.logger.warning(f"Failed to fetch backtest metrics for {symbol}: {e}")
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.5,
                'max_drawdown': 0.0,
                'profit_factor': 1.0,
                'avg_trade_return': 0.0,
                'trade_count': 0,
            }

    def _score_backtest(self, metrics: Dict) -> float:
        """
        Score based on backtest performance metrics.

        Combines:
        - Sharpe ratio (risk-adjusted return)
        - Win rate
        - Profit factor
        - Drawdown (negative impact)

        Args:
            metrics: Dict with backtest metrics

        Returns:
            Score from 0 to 100
        """
        if metrics.get('trade_count', 0) < 10:
            return 50.0  # Neutral if insufficient data

        # Component scores (each 0-100)
        # Sharpe: 0 = 0, 1 = 50, 2+ = 100
        sharpe = metrics.get('sharpe_ratio', 0)
        sharpe_score = min(100, max(0, sharpe * 50))

        # Win rate: 30% = 0, 50% = 50, 70%+ = 100
        win_rate = metrics.get('win_rate', 0.5)
        win_score = min(100, max(0, (win_rate - 0.3) / 0.4 * 100))

        # Profit factor: 0.5 = 0, 1.0 = 30, 2.0 = 100
        pf = metrics.get('profit_factor', 1.0)
        if pf <= 0.5:
            pf_score = 0
        elif pf <= 1.0:
            pf_score = (pf - 0.5) / 0.5 * 30
        else:
            pf_score = 30 + min(70, (pf - 1.0) * 70)

        # Drawdown: 0% = 100, 5% = 70, 10%+ = 0
        dd = metrics.get('max_drawdown', 0)
        dd_score = max(0, 100 - dd * 1000)  # 10% = 0

        # Weighted combination
        backtest_score = (
            0.35 * sharpe_score +
            0.25 * win_score +
            0.25 * pf_score +
            0.15 * dd_score
        )

        return backtest_score

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
        print("\n" + "=" * 100)
        print("PAIR SCORING REPORT (Multi-Factor)")
        print("=" * 100)
        print(f"{'Symbol':<12} {'Score':>6} {'Rec':>10} {'Volume':>10} {'Spread':>7} {'Sharpe':>7} {'WinRate':>8} {'Backtest':>8} {'ML':>6}")
        print("-" * 100)

        for s in scores:
            vol_str = f"${s.volume_24h/1_000_000:.1f}M"
            sharpe_str = f"{s.sharpe_ratio:.2f}"
            win_str = f"{s.win_rate*100:.1f}%"
            bt_str = f"{s.backtest_score:.0f}"
            ml_str = f"{s.ml_score:.0f}" if s.ml_score is not None else "N/A"
            print(f"{s.symbol:<12} {s.composite_score:>6.1f} {s.recommendation:>10} {vol_str:>10} {s.spread_pct:>6.3f}% {sharpe_str:>7} {win_str:>8} {bt_str:>8} {ml_str:>6}")

        print("=" * 100)
        print("\nScore Breakdown (weights):")
        weight_str = " | ".join([
            f"Volume: {self.weights['volume']*100:.0f}%",
            f"Spread: {self.weights['spread']*100:.0f}%",
            f"Fees: {self.weights['fees']*100:.0f}%",
            f"Fill: {self.weights['fill_rate']*100:.0f}%",
            f"Backtest: {self.weights['backtest']*100:.0f}%",
        ])
        print(f"  {weight_str}")
        if self.ml_enabled:
            print(f"  ML Weight in Ensemble: {self.ml_weight*100:.0f}%")
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
