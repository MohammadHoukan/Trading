"""
Grid parameter optimizer.
Runs parameter sweeps to find optimal settings for a given period.
"""
import argparse
import sys
import os
import logging
import json
import itertools
import pandas as pd
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.data_fetcher import fetch_ohlcv
from backtest.simulator import GridSimulator
from backtest.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize(
    symbol: str, 
    days: int, 
    capital: float,
    rolling: bool = True,
    save: bool = False
):
    print(f"\n🧪 Optimizing {symbol} over {days} days (Rolling={rolling})...")
    
    # Fetch Data
    ohlcv = fetch_ohlcv(symbol, days=days)
    current_price = ohlcv['close'].iloc[-1]
    mean_price = ohlcv['close'].mean()
    high_price = ohlcv['high'].max()
    low_price = ohlcv['low'].min()
    
    print(f"   Price Range: ${low_price:.2f} - ${high_price:.2f}")
    
    # Parameter Space
    grid_levels_options = [10, 20, 30, 40, 50, 60, 80, 100]
    range_multipliers = [0.8, 1.0, 1.2, 1.5, 2.0] # Relative to actual High-Low range
    
    results = []
    
    # Grid Search
    total_iterations = len(grid_levels_options) * len(range_multipliers)
    print(f"   Running {total_iterations} combinations...")
    
    count = 0
    for grid_levels in grid_levels_options:
        for range_mult in range_multipliers:
            count += 1
            if count % 10 == 0:
                print(f"   ... {count}/{total_iterations}")
            
            # Calculate range based on multiplier
            center = mean_price
            price_spread = (high_price - low_price) * range_mult
            # Ensure minimal spread
            if price_spread < center * 0.05:
                price_spread = center * 0.05
                
            lower = center - (price_spread / 2)
            upper = center + (price_spread / 2)
            
            # Ensure lower is positive
            if lower <= 0:
                lower = low_price * 0.5
            
            # Calculate amount per grid (simple logic: capital / levels / price)
            # This is an approximation. In reality, amount is base currency.
            # cost per grid ~ price * amount
            # total investment ~ levels * price * amount
            # amount ~ capital / (levels * price)
            amount = (capital * 0.95) / (grid_levels * center)
            
            # Stop loss just below lower limit
            stop_loss = lower * 0.95
            
            try:
                simulator = GridSimulator(
                    lower_limit=lower,
                    upper_limit=upper,
                    grid_levels=grid_levels,
                    amount_per_grid=amount,
                    initial_capital=capital,
                    stop_loss=stop_loss,
                    rolling=rolling
                )
                
                result = simulator.run(ohlcv)
                metrics = calculate_metrics(
                    result.trades,
                    result.initial_capital,
                    result.final_capital,
                    result.equity_curve
                )
                
                results.append({
                    'grids': grid_levels,
                    'range_mult': range_mult,
                    'lower': lower,
                    'upper': upper,
                    'amount': amount,
                    'pnl': metrics.total_pnl,
                    'return_pct': metrics.return_pct,
                    'sharpe': metrics.sharpe_ratio,
                    'trades': metrics.num_trades,
                    'drawdown': metrics.max_drawdown_pct
                })
                
            except Exception as e:
                pass
                
    # Sort by PnL
    df = pd.DataFrame(results)
    df = df.sort_values('pnl', ascending=False)
    
    best = df.iloc[0]
    
    print("\n🏆 OPTIMIZATION RESULTS (Top 5)")
    print(df[['grids', 'range_mult', 'return_pct', 'drawdown', 'trades', 'sharpe']].head(5).to_string(index=False))
    
    print(f"\n✨ Best Settings for {symbol}:")
    print(f"   Grid Levels: {int(best.grids)}")
    print(f"   Range: ${best.lower:.2f} - ${best.upper:.2f}")
    print(f"   Amount: {best.amount:.4f}")
    print(f"   Return: {best.return_pct:.2f}% (${best.pnl:.2f})")
    print(f"   Drawdown: {best.drawdown:.2f}%")
    
    if save:
        update_strategy_config(symbol, best, rolling)
    
    return best


def update_strategy_config(symbol: str, best_params: pd.Series, rolling: bool):
    """Update strategies.json with optimized parameters."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategies.json')
    
    try:
        with open(config_path, 'r') as f:
            strategies = json.load(f)
            
        if symbol not in strategies:
            strategies[symbol] = {"enabled": True}
            
        # Update params
        strategies[symbol].update({
            "grid_levels": int(best_params.grids),
            "lower_limit": float(f"{best_params.lower:.2f}"),
            "upper_limit": float(f"{best_params.upper:.2f}"),
            "amount_per_grid": float(f"{best_params.amount:.4f}"),
            "rolling_grids": rolling,
            # Set stop loss 5% below lower limit
            "stop_loss": float(f"{best_params.lower * 0.95:.2f}")
        })
        
        with open(config_path, 'w') as f:
            json.dump(strategies, f, indent=4)
            
        print(f"\n💾 Saved optimized settings to config/strategies.json")
        
    except Exception as e:
        logger.error(f"Failed to update config: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--save', action='store_true', help='Update strategies.json with best results')
    args = parser.parse_args()
    
    optimize(args.pair, args.days, args.capital, save=args.save)
