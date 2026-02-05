
import pandas as pd
import numpy as np
import sys
import os

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtest.simulator import GridSimulator
from backtest.execution_model import ExecutionModel
from datetime import datetime, timedelta

def generate_mock_data(days=5, volatility=0.01):
    """Generate mock trending/ranging data for testing."""
    np.random.seed(42)
    periods = days * 24 * 4  # 15m intervals
    
    # Ranging market
    prices = [100.0]
    for _ in range(periods):
        change = np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame(prices, columns=['close'])
    df['open'] = df['close'].shift(1).fillna(100.0)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + 0.002)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - 0.002)
    df.index = [datetime.now() + timedelta(minutes=15*i) for i in range(len(df))]
    return df

def run_test():
    df = generate_mock_data()
    
    # Grid Params
    lower, upper = 90, 110
    levels = 20
    amount = 1.0
    
    print("=== Backtest Comparison ===")
    
    # 1. Ideal Execution
    sim_ideal = GridSimulator(lower, upper, levels, amount, initial_capital=10000)
    res_ideal = sim_ideal.run(df)
    
    # 2. Realistic Execution
    em = ExecutionModel(
        slippage_bps=10, 
        fill_probability=0.5, 
        spread_bps=10,
        enabled=True
    )
    sim_real = GridSimulator(lower, upper, levels, amount, initial_capital=10000, execution_model=em)
    res_real = sim_real.run(df)
    
    print(f"\nIdeal Results:")
    print(f"  Trades: {len(res_ideal.trades)}")
    print(f"  PnL:    ${res_ideal.total_pnl:.2f}")
    
    print(f"\nRealistic Results:")
    print(f"  Trades: {len(res_real.trades)}")
    print(f"  PnL:    ${res_real.total_pnl:.2f}")
    
    drag = res_ideal.total_pnl - res_real.total_pnl
    print(f"\nExecution Drag: ${drag:.2f} ({ (drag/res_ideal.total_pnl)*100 if res_ideal.total_pnl != 0 else 0 :.1f}%)")
    
    if len(res_real.trades) < len(res_ideal.trades):
        print("\nSUCCESS: Realistic model reduced trade count (partial fills/missed fills).")
    if res_real.total_pnl < res_ideal.total_pnl:
        print("SUCCESS: Realistic model reduced PnL (slippage/spread).")

if __name__ == "__main__":
    run_test()
