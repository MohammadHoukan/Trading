"""
Portfolio backtester for running multiple grid bots concurrently.

Usage:
    python -m backtest.portfolio_runner --days 30 --capital 1000
"""
import argparse
import sys
import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.data_fetcher import fetch_ohlcv
from backtest.simulator import GridSimulator, BacktestResult
from backtest.metrics import calculate_metrics, format_metrics, Metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_strategies() -> Dict[str, dict]:
    """Load all enabled strategies from strategies.json."""
    strat_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategies.json')
    try:
        with open(strat_path, 'r') as f:
            strategies = json.load(f)
            # Filter for enabled strategies only
            enabled = {
                pair: conf for pair, conf in strategies.items() 
                if conf.get('enabled', False)
            }
            return enabled
    except Exception as e:
        logger.error(f"Could not load strategies: {e}")
        return {}


def run_portfolio_backtest(
    days: int,
    initial_capital_per_bot: float,
    fees_percent: float = 0.1
):
    """Run backtest for all enabled strategies."""
    strategies = load_all_strategies()
    if not strategies:
        print("❌ No enabled strategies found in config/strategies.json")
        return

    print(f"\n🚀 Starting Portfolio Backtest ({days} days)")
    print(f"   Strategies: {', '.join(strategies.keys())}")
    print(f"   Capital per Bot: ${initial_capital_per_bot:.2f}")
    
    results: Dict[str, BacktestResult] = {}
    metrics_map: Dict[str, Metrics] = {}
    
    # 1. Run Simulations
    for pair, config in strategies.items():
        print(f"\nTesting {pair}...")
        
        try:
            # Fetch data
            ohlcv = fetch_ohlcv(pair, timeframe='1h', days=days)
            
            # Setup Simulator
            simulator = GridSimulator(
                lower_limit=config['lower_limit'],
                upper_limit=config['upper_limit'],
                grid_levels=config['grid_levels'],
                amount_per_grid=config['amount_per_grid'],
                initial_capital=initial_capital_per_bot,
                stop_loss=config.get('stop_loss'),
                fees_percent=fees_percent,
                rolling=config.get('rolling_grids', False)
            )
            
            # Run
            result = simulator.run(ohlcv)
            results[pair] = result
            
            # Calc Metrics
            metrics = calculate_metrics(
                result.trades,
                result.initial_capital,
                result.final_capital,
                result.equity_curve
            )
            metrics_map[pair] = metrics
            
            print(f"   PnL: ${metrics.total_pnl:.2f} ({metrics.return_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to test {pair}: {e}")
            
    # 2. Portfolio Aggregation
    print("\n╔══════════════════════════════════════════════════╗")
    print("║            PORTFOLIO REPORT                      ║")
    print("╠══════════════════════════════════════════════════╣")
    
    total_capital = initial_capital_per_bot * len(results)
    final_portfolio_value = total_capital
    portfolio_pnl = 0.0
    
    # Aggregate equity curves
    # We need to align timestamps. A robust way is to reindex to a common range.
    # For simplicity, we assume 1h candles align mostly well. 
    # We'll sum available equity curves.
    
    combined_equity = pd.Series(dtype=float)
    
    for pair, res in results.items():
        if res.equity_curve.empty:
            continue
            
        if combined_equity.empty:
            combined_equity = res.equity_curve.copy()
        else:
            # Add to existing, filling missing with 0 (should use intelligent fill in production)
            # Better: Only sum overlapping periods or enforce same timeframe fetching
            combined_equity = combined_equity.add(res.equity_curve, fill_value=0)
            
        portfolio_pnl += res.total_pnl

    final_portfolio_value += portfolio_pnl
    portfolio_return = (portfolio_pnl / total_capital) * 100 if total_capital > 0 else 0.0
    
    print(f"║  Total Capital:    ${total_capital:>12,.2f}              ║")
    print(f"║  Final Value:      ${final_portfolio_value:>12,.2f}              ║")
    print(f"║  Total PnL:        ${portfolio_pnl:>12,.2f}              ║")
    print(f"║  Return:           {portfolio_return:>12.2f}%              ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Breakdown:                                      ║")
    
    for pair, m in metrics_map.items():
        print(f"║  {pair:<10} ${m.total_pnl:>9.2f} ({m.return_pct:>6.2f}%)       ║")
        
    print("╚══════════════════════════════════════════════════╝")
    
    # 3. Correlation Matrix
    print("\n🔗 Correlation Matrix (Daily Returns):")
    returns_df = pd.DataFrame()
    for pair, res in results.items():
        if not res.equity_curve.empty:
            # Resample to daily for correlation
            daily_equity = res.equity_curve.resample('D').last().ffill()
            daily_returns = daily_equity.pct_change().fillna(0)
            returns_df[pair] = daily_returns
            
    if not returns_df.empty:
        corr_matrix = returns_df.corr()
        print(corr_matrix.round(2))
    
    # 4. Visualization
    print("\n🎨 Generating Portfolio Plot...")
    plt.figure(figsize=(12, 8))
    
    # Plot individual equities (normalized to start)
    for pair, res in results.items():
        if not res.equity_curve.empty:
            plt.plot(res.equity_curve.index, res.equity_curve, label=f"{pair} ({metrics_map[pair].return_pct:.1f}%)", alpha=0.6)
            
    # Plot Portfolio
    plt.plot(combined_equity.index, combined_equity, label=f"Portfolio ({portfolio_return:.1f}%)", linewidth=2.5, color='black')
    
    plt.title(f"Portfolio Performance (Rollling Grid) - {days} Days")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = 'portfolio_performance.png'
    plt.savefig(output_file)
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio Backtester')
    parser.add_argument('--days', '-d', type=int, default=30, help='Days of history')
    parser.add_argument('--capital', '-c', type=float, default=1000.0, help='Capital per bot')
    
    args = parser.parse_args()
    
    run_portfolio_backtest(args.days, args.capital)
