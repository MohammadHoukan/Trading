"""
Regime Filter Validation Script

Validates whether the regime filter strategy actually saves money by comparing:
1. "Always Trade" - continuous grid trading, ignoring market regime
2. "Regime Filtered" - pauses during TRENDING periods, trades during RANGING

This answers the key question: "Does pausing during unfavorable regimes improve profitability?"

Usage:
    python -m backtest.regime_validator --pair SOL/USDT --days 60
    python -m backtest.regime_validator --pair ETH/USDT --days 90 --verbose
"""

import argparse
import sys
import os
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.data_fetcher import fetch_ohlcv
from backtest.simulator import GridSimulator, BacktestResult
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RegimeSegment:
    """A continuous period of a specific regime."""
    start: pd.Timestamp
    end: pd.Timestamp
    regime: str  # 'RANGING', 'TRENDING', 'UNCERTAIN'
    score: float
    duration_hours: float


@dataclass
class ValidationResult:
    """Results from regime validation."""
    symbol: str
    days: int

    # Regime statistics
    total_hours: float
    ranging_hours: float
    trending_hours: float
    uncertain_hours: float

    # Always-trade results
    always_trade_pnl: float
    always_trade_return_pct: float
    always_trade_trades: int

    # Regime-filtered results
    filtered_pnl: float
    filtered_return_pct: float
    filtered_trades: int

    # Per-regime analysis
    ranging_pnl: float  # PnL from trading during RANGING
    trending_pnl_if_traded: float  # Hypothetical PnL if we traded during TRENDING

    # Key metrics
    pnl_saved: float  # filtered_pnl - always_trade_pnl
    pnl_saved_pct: float
    regime_filter_effective: bool  # Did filtering improve results?


class RegimeCalculator:
    """
    Simplified regime calculation for backtesting.

    Uses the same signals as RegimeFilter but operates on historical data
    without requiring live exchange connection.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        regime_cfg = self.config.get('regime', {})

        # Thresholds (same as RegimeFilter)
        self.adx_threshold = regime_cfg.get('adx_threshold', 30.0)
        self.volatility_threshold = regime_cfg.get('volatility_threshold', 0.03)
        self.ma_distance_threshold = regime_cfg.get('ma_distance_threshold', 0.02)

        # Weights (same as RegimeFilter)
        self.weights = {
            'adx': 0.40,  # Increased since we don't have fill rate
            'volatility': 0.30,
            'ma_distance': 0.30,
        }

    def calculate_regimes(self, ohlcv_df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """
        Calculate regime for each candle in the dataset.

        Args:
            ohlcv_df: OHLCV DataFrame with datetime index
            lookback: Number of candles for indicator calculation

        Returns:
            DataFrame with columns: regime, score, adx, volatility, ma_distance
        """
        try:
            import pandas_ta as ta
        except ImportError:
            logger.error("pandas_ta not installed. Run: pip install pandas_ta")
            raise

        df = ohlcv_df.copy()

        # Calculate indicators
        # ADX
        adx_df = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)
        adx_col = next((c for c in adx_df.columns if str(c).startswith('ADX_')), None)
        df['adx'] = adx_df[adx_col] if adx_col else np.nan

        # ATR / Price (volatility)
        atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
        df['volatility'] = atr / df['close']

        # MA Distance
        df['sma_50'] = df['close'].rolling(lookback).mean()
        df['ma_distance'] = (df['close'] - df['sma_50']) / df['sma_50']

        # Calculate composite score for each row
        def calc_score(row):
            scores = {}

            # ADX: Lower is better for grid (ranging = good)
            if pd.notna(row['adx']):
                adx_score = max(0, 100 - (row['adx'] / self.adx_threshold) * 50)
                scores['adx'] = min(100, adx_score)

            # Volatility: Moderate is best
            if pd.notna(row['volatility']):
                vol_ratio = row['volatility'] / self.volatility_threshold
                if vol_ratio < 0.5:
                    vol_score = 50  # Too low
                elif vol_ratio < 1.5:
                    vol_score = 100  # Sweet spot
                else:
                    vol_score = max(0, 100 - (vol_ratio - 1.5) * 50)
                scores['volatility'] = vol_score

            # MA Distance: Closer to MA = better
            if pd.notna(row['ma_distance']):
                abs_dist = abs(row['ma_distance'])
                ma_score = max(0, 100 - (abs_dist / self.ma_distance_threshold) * 50)
                scores['ma_distance'] = min(100, ma_score)

            if not scores:
                return 50.0  # Neutral if no data

            total_weight = sum(self.weights[k] for k in scores.keys())
            return sum(scores[k] * self.weights[k] for k in scores.keys()) / total_weight

        df['score'] = df.apply(calc_score, axis=1)

        # Determine regime
        def get_regime(score):
            if score >= 60:
                return 'RANGING'
            elif score >= 40:
                return 'UNCERTAIN'
            else:
                return 'TRENDING'

        df['regime'] = df['score'].apply(get_regime)

        return df[['regime', 'score', 'adx', 'volatility', 'ma_distance']]


def segment_regimes(regime_df: pd.DataFrame) -> List[RegimeSegment]:
    """Convert regime series into continuous segments."""
    segments = []
    current_regime = None
    segment_start = None
    scores = []

    for timestamp, row in regime_df.iterrows():
        if current_regime != row['regime']:
            # Close previous segment
            if current_regime is not None:
                duration = (timestamp - segment_start).total_seconds() / 3600
                segments.append(RegimeSegment(
                    start=segment_start,
                    end=timestamp,
                    regime=current_regime,
                    score=np.mean(scores),
                    duration_hours=duration
                ))

            # Start new segment
            current_regime = row['regime']
            segment_start = timestamp
            scores = [row['score']]
        else:
            scores.append(row['score'])

    # Close final segment
    if current_regime is not None:
        final_ts = regime_df.index[-1]
        duration = (final_ts - segment_start).total_seconds() / 3600
        segments.append(RegimeSegment(
            start=segment_start,
            end=final_ts,
            regime=current_regime,
            score=np.mean(scores),
            duration_hours=duration
        ))

    return segments


def run_simulation_for_period(
    ohlcv_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    grid_params: dict,
    initial_inventory: float = 0.0,
    initial_avg_cost: float = 0.0,
    initial_capital: float = 1000.0
) -> Tuple[float, int, float, float, float]:
    """
    Run grid simulation for a specific time period.

    Returns:
        (pnl, trade_count, final_inventory, final_avg_cost, final_capital)
    """
    period_df = ohlcv_df.loc[start:end]
    if len(period_df) < 2:
        return 0.0, 0, initial_inventory, initial_avg_cost, initial_capital

    # Create simulator with current state
    simulator = GridSimulator(
        lower_limit=grid_params['lower_limit'],
        upper_limit=grid_params['upper_limit'],
        grid_levels=grid_params['grid_levels'],
        amount_per_grid=grid_params['amount_per_grid'],
        initial_capital=initial_capital,
        fees_percent=grid_params.get('fees_percent', 0.1),
        rolling=grid_params.get('rolling', False)
    )

    result = simulator.run(period_df)
    return (
        result.total_pnl,
        len(result.trades),
        result.final_inventory,
        result.final_avg_cost,
        result.final_capital
    )


class RegimeAwareSimulator:
    """
    Grid simulator that respects regime signals.

    Pauses trading during TRENDING periods while maintaining grid state.
    """

    def __init__(
        self,
        lower_limit: float,
        upper_limit: float,
        grid_levels: int,
        amount_per_grid: float,
        initial_capital: float = 1000.0,
        fees_percent: float = 0.1,
        rolling: bool = False
    ):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.grid_levels = grid_levels
        self.amount_per_grid = amount_per_grid
        self.initial_capital = initial_capital
        self.fees_percent = fees_percent
        self.rolling = rolling

        # Calculate grid
        self.grid_step = (upper_limit - lower_limit) / grid_levels
        self.grid_prices = [lower_limit + i * self.grid_step for i in range(grid_levels + 1)]

    def run(self, ohlcv_df: pd.DataFrame, regime_df: pd.DataFrame) -> Tuple[BacktestResult, Dict]:
        """
        Run backtest respecting regime signals.

        Args:
            ohlcv_df: OHLCV data
            regime_df: DataFrame with 'regime' column

        Returns:
            (BacktestResult, stats_by_regime)
        """
        # State
        inventory = 0.0
        avg_cost = 0.0
        realized_pnl = 0.0
        capital = self.initial_capital

        buy_levels = set()
        sell_levels = set()

        trades = []
        equity_curve = []

        # Stats by regime
        stats = {
            'RANGING': {'pnl': 0.0, 'trades': 0, 'candles': 0},
            'TRENDING': {'pnl': 0.0, 'trades': 0, 'candles': 0},
            'UNCERTAIN': {'pnl': 0.0, 'trades': 0, 'candles': 0},
        }

        # Initialize grid
        first_price = ohlcv_df['close'].iloc[0]
        for i, price in enumerate(self.grid_prices):
            if price < first_price * 0.995:
                buy_levels.add(i)
            elif price > first_price * 1.005:
                sell_levels.add(i)

        # Simulate
        for timestamp, row in ohlcv_df.iterrows():
            regime = regime_df.loc[timestamp, 'regime'] if timestamp in regime_df.index else 'UNCERTAIN'
            high = row['high']
            low = row['low']
            close = row['close']

            stats[regime]['candles'] += 1
            period_start_pnl = realized_pnl

            # Skip trading during TRENDING
            if regime == 'TRENDING':
                # Track equity but don't trade
                unrealized = inventory * close - (avg_cost * inventory) if inventory > 0 else 0.0
                equity = capital + (inventory * close)
                equity_curve.append({'timestamp': timestamp, 'equity': equity, 'price': close, 'regime': regime})
                continue

            # Trading logic (same as GridSimulator)
            phases = ['buy', 'sell'] if row['open'] <= row['close'] else ['sell', 'buy']

            for phase in phases:
                if phase == 'buy':
                    for level in sorted(buy_levels, reverse=True):
                        if level not in buy_levels or level >= len(self.grid_prices):
                            continue
                        grid_price = self.grid_prices[level]
                        if low <= grid_price:
                            cost = grid_price * self.amount_per_grid
                            fee = cost * (self.fees_percent / 100)

                            if capital >= cost + fee:
                                capital -= cost + fee
                                total_cost = (avg_cost * inventory) + cost + fee
                                inventory += self.amount_per_grid
                                avg_cost = total_cost / inventory if inventory > 0 else 0.0

                                trades.append({
                                    'timestamp': timestamp,
                                    'side': 'buy',
                                    'price': grid_price,
                                    'amount': self.amount_per_grid,
                                    'grid_level': level,
                                    'regime': regime
                                })
                                buy_levels.discard(level)

                                if level + 1 <= self.grid_levels:
                                    sell_levels.add(level + 1)

                elif phase == 'sell':
                    for level in sorted(sell_levels):
                        if level not in sell_levels or level >= len(self.grid_prices):
                            continue
                        grid_price = self.grid_prices[level]
                        if high >= grid_price and inventory > 0:
                            sell_amount = min(self.amount_per_grid, inventory)
                            proceeds = grid_price * sell_amount
                            fee = proceeds * (self.fees_percent / 100)

                            pnl = (grid_price - avg_cost) * sell_amount - fee
                            realized_pnl += pnl
                            capital += proceeds - fee
                            inventory -= sell_amount

                            if inventory <= 0:
                                inventory = 0.0
                                avg_cost = 0.0

                            trades.append({
                                'timestamp': timestamp,
                                'side': 'sell',
                                'price': grid_price,
                                'amount': sell_amount,
                                'grid_level': level,
                                'pnl': pnl,
                                'regime': regime
                            })
                            sell_levels.discard(level)

                            if level - 1 >= 0:
                                buy_levels.add(level - 1)

            # Track regime stats
            period_pnl = realized_pnl - period_start_pnl
            stats[regime]['pnl'] += period_pnl
            stats[regime]['trades'] += sum(1 for t in trades if t.get('timestamp') == timestamp)

            # Track equity
            unrealized = inventory * close - (avg_cost * inventory) if inventory > 0 else 0.0
            equity = capital + (inventory * close)
            equity_curve.append({'timestamp': timestamp, 'equity': equity, 'price': close, 'regime': regime})

        # Final calculations
        final_close = ohlcv_df['close'].iloc[-1]
        unrealized_pnl = (final_close - avg_cost) * inventory if inventory > 0 else 0.0
        total_pnl = realized_pnl + unrealized_pnl
        final_capital = capital + (inventory * final_close)

        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)

        result = BacktestResult(
            trades=[],  # Simplified
            final_inventory=inventory,
            final_avg_cost=avg_cost,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            stop_loss_triggered=False,
            stop_loss_price=None,
            price_series=ohlcv_df['close'],
            equity_curve=equity_df['equity'] if not equity_df.empty else pd.Series()
        )

        return result, stats, trades


def validate_regime_filter(
    symbol: str,
    days: int = 60,
    timeframe: str = '1h',
    grid_params: dict = None,
    initial_capital: float = 1000.0,
    verbose: bool = False
) -> ValidationResult:
    """
    Run comprehensive regime filter validation.

    Args:
        symbol: Trading pair
        days: Days of historical data
        timeframe: Candle timeframe
        grid_params: Grid strategy parameters (auto-calculated if None)
        initial_capital: Starting capital
        verbose: Print detailed output

    Returns:
        ValidationResult with comparison metrics
    """
    print(f"\n{'='*70}")
    print(f"REGIME FILTER VALIDATION: {symbol}")
    print(f"{'='*70}")

    # 1. Fetch data
    print(f"\n Fetching {days} days of {timeframe} data...")
    ohlcv = fetch_ohlcv(symbol, timeframe=timeframe, days=days)
    print(f"   Loaded {len(ohlcv)} candles from {ohlcv.index[0]} to {ohlcv.index[-1]}")

    # 2. Calculate regimes
    print(f"\n Calculating market regimes...")
    calc = RegimeCalculator()
    regime_df = calc.calculate_regimes(ohlcv)

    # 3. Get regime segments for reporting
    segments = segment_regimes(regime_df)

    # Calculate regime distribution
    total_hours = len(ohlcv)  # Each candle = 1 hour for 1h timeframe
    ranging_count = (regime_df['regime'] == 'RANGING').sum()
    trending_count = (regime_df['regime'] == 'TRENDING').sum()
    uncertain_count = (regime_df['regime'] == 'UNCERTAIN').sum()

    print(f"\n Regime Distribution:")
    print(f"   RANGING:   {ranging_count:>6} candles ({ranging_count/len(regime_df)*100:>5.1f}%)")
    print(f"   TRENDING:  {trending_count:>6} candles ({trending_count/len(regime_df)*100:>5.1f}%)")
    print(f"   UNCERTAIN: {uncertain_count:>6} candles ({uncertain_count/len(regime_df)*100:>5.1f}%)")

    # 4. Auto-calculate grid params if not provided
    if grid_params is None:
        first_price = ohlcv['close'].iloc[0]

        # Grid centered around first price with reasonable range
        # Use 15% above and below for typical crypto volatility
        grid_range_pct = 0.15

        # Calculate amount per grid based on capital and price
        # Each grid uses ~2% of capital
        amount_per_grid = (initial_capital * 0.02) / first_price

        grid_params = {
            'lower_limit': first_price * (1 - grid_range_pct),
            'upper_limit': first_price * (1 + grid_range_pct),
            'grid_levels': 20,
            'amount_per_grid': amount_per_grid,
            'fees_percent': 0.1,
            'rolling': True  # Enable rolling to handle price breakouts
        }

    print(f"\n Grid Parameters:")
    print(f"   Range: ${grid_params['lower_limit']:.2f} - ${grid_params['upper_limit']:.2f}")
    print(f"   Levels: {grid_params['grid_levels']}")
    print(f"   Amount/Grid: {grid_params['amount_per_grid']:.4f}")

    # 5. Run "Always Trade" simulation (standard GridSimulator)
    print(f"\n Running ALWAYS TRADE simulation...")
    always_sim = GridSimulator(
        lower_limit=grid_params['lower_limit'],
        upper_limit=grid_params['upper_limit'],
        grid_levels=grid_params['grid_levels'],
        amount_per_grid=grid_params['amount_per_grid'],
        initial_capital=initial_capital,
        fees_percent=grid_params.get('fees_percent', 0.1),
        rolling=grid_params.get('rolling', False)
    )
    always_result = always_sim.run(ohlcv)

    # 6. Run "Regime Filtered" simulation
    print(f" Running REGIME FILTERED simulation...")
    filtered_sim = RegimeAwareSimulator(
        lower_limit=grid_params['lower_limit'],
        upper_limit=grid_params['upper_limit'],
        grid_levels=grid_params['grid_levels'],
        amount_per_grid=grid_params['amount_per_grid'],
        initial_capital=initial_capital,
        fees_percent=grid_params.get('fees_percent', 0.1),
        rolling=grid_params.get('rolling', False)
    )
    filtered_result, regime_stats, filtered_trades = filtered_sim.run(ohlcv, regime_df)

    # 7. Calculate "what if we traded during TRENDING" by looking at always_result
    # Segment the always_result trades by regime
    always_trades_by_regime = {'RANGING': [], 'TRENDING': [], 'UNCERTAIN': []}
    for trade in always_result.trades:
        ts = trade.timestamp
        if ts in regime_df.index:
            regime = regime_df.loc[ts, 'regime']
            always_trades_by_regime[regime].append(trade)

    trending_hypothetical_pnl = sum(t.pnl for t in always_trades_by_regime['TRENDING'])
    ranging_pnl_always = sum(t.pnl for t in always_trades_by_regime['RANGING'])

    if verbose:
        print(f"\n Trades by Regime (Always Trade):")
        for regime, trades in always_trades_by_regime.items():
            pnl = sum(t.pnl for t in trades)
            print(f"   {regime}: {len(trades)} trades, PnL=${pnl:.2f}")

        print(f"\n Regime Stats (Filtered):")
        for regime, stats in regime_stats.items():
            print(f"   {regime}: {stats['candles']} candles, {stats['trades']} trades, PnL=${stats['pnl']:.2f}")

    # 8. Calculate results
    always_return_pct = (always_result.final_capital - initial_capital) / initial_capital * 100
    filtered_return_pct = (filtered_result.final_capital - initial_capital) / initial_capital * 100

    pnl_saved = filtered_result.total_pnl - always_result.total_pnl
    pnl_saved_pct = (pnl_saved / abs(always_result.total_pnl) * 100) if always_result.total_pnl != 0 else 0

    ranging_pnl = regime_stats['RANGING']['pnl'] + regime_stats['UNCERTAIN']['pnl']
    filtered_trades_count = len(filtered_trades)

    # 9. Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")

    print(f"\n ALWAYS TRADE (Baseline):")
    print(f"   Total PnL:      ${always_result.total_pnl:>10.2f}")
    print(f"   Return:         {always_return_pct:>10.2f}%")
    print(f"   Trades:         {len(always_result.trades):>10}")
    print(f"   Final Capital:  ${always_result.final_capital:>10.2f}")

    print(f"\n REGIME FILTERED:")
    print(f"   Total PnL:      ${filtered_result.total_pnl:>10.2f}")
    print(f"   Return:         {filtered_return_pct:>10.2f}%")
    print(f"   Trades:         {filtered_trades_count:>10}")
    print(f"   Final Capital:  ${filtered_result.final_capital:>10.2f}")

    print(f"\n REGIME BREAKDOWN (from Always Trade):")
    print(f"   RANGING trades:  {len(always_trades_by_regime['RANGING']):>4}, PnL=${ranging_pnl_always:>8.2f}")
    print(f"   TRENDING trades: {len(always_trades_by_regime['TRENDING']):>4}, PnL=${trending_hypothetical_pnl:>8.2f}")
    print(f"   UNCERTAIN trades:{len(always_trades_by_regime['UNCERTAIN']):>4}")

    print(f"\n CONCLUSION:")
    print(f"   PnL Difference: ${pnl_saved:>10.2f}")

    regime_filter_effective = filtered_result.total_pnl > always_result.total_pnl

    if regime_filter_effective:
        print(f"   REGIME FILTER WORKS! Saved ${abs(pnl_saved):.2f} by pausing during TRENDING")
        if trending_hypothetical_pnl < 0:
            print(f"   Trading during TRENDING would have LOST ${abs(trending_hypothetical_pnl):.2f}")
    else:
        print(f"   REGIME FILTER UNDERPERFORMED by ${abs(pnl_saved):.2f}")
        if trending_hypothetical_pnl > 0:
            print(f"   Missed ${trending_hypothetical_pnl:.2f} profit by pausing during TRENDING")

    # Edge case analysis
    if trending_hypothetical_pnl < 0:
        print(f"\n   Key Insight: TRENDING periods were net NEGATIVE (${trending_hypothetical_pnl:.2f})")
        print(f"   => Pausing during trends is a valid strategy for this period!")
    elif trending_hypothetical_pnl > 0:
        print(f"\n   Key Insight: TRENDING periods were net POSITIVE (${trending_hypothetical_pnl:.2f})")
        print(f"   => Consider adjusting regime thresholds or allowing some trend trading.")

    print(f"\n{'='*70}\n")

    # Convert hours for result
    ranging_hours = ranging_count  # Assuming 1h timeframe
    trending_hours = trending_count
    uncertain_hours = uncertain_count

    return ValidationResult(
        symbol=symbol,
        days=days,
        total_hours=float(total_hours),
        ranging_hours=float(ranging_hours),
        trending_hours=float(trending_hours),
        uncertain_hours=float(uncertain_hours),
        always_trade_pnl=always_result.total_pnl,
        always_trade_return_pct=always_return_pct,
        always_trade_trades=len(always_result.trades),
        filtered_pnl=filtered_result.total_pnl,
        filtered_return_pct=filtered_return_pct,
        filtered_trades=filtered_trades_count,
        ranging_pnl=ranging_pnl,
        trending_pnl_if_traded=trending_hypothetical_pnl,
        pnl_saved=pnl_saved,
        pnl_saved_pct=pnl_saved_pct,
        regime_filter_effective=regime_filter_effective
    )


def main():
    parser = argparse.ArgumentParser(description='Regime Filter Validation')
    parser.add_argument('--pair', '-p', default='SOL/USDT', help='Trading pair')
    parser.add_argument('--days', '-d', type=int, default=60, help='Days of history')
    parser.add_argument('--timeframe', '-t', default='1h', help='Candle timeframe')
    parser.add_argument('--capital', '-c', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--verbose', '-v', action='store_true', help='Detailed output')

    args = parser.parse_args()

    result = validate_regime_filter(
        symbol=args.pair,
        days=args.days,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        verbose=args.verbose
    )

    return result


if __name__ == '__main__':
    main()
