import pandas as pd
import numpy as np
import sys
import os

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtest.simulator import GridSimulator

def test_rolling_grid_logic():
    print("Testing Rolling Grid Logic...")
    
    # Setup simulator with rolling grids
    sim = GridSimulator(
        lower_limit=100.0,
        upper_limit=110.0,
        grid_levels=10,
        amount_per_grid=1.0,
        initial_capital=10000.0,
        rolling=True
    )
    
    # Complex price path:
    # 1. Start at center (105)
    # 2. Break top (112)
    # 3. Oscillate at new high (110-114)
    # 4. Crash through bottom (114 -> 95)
    # 5. Recovery (95 -> 105)
    prices = [105, 107, 109, 111, 112, 113, 111, 114, 110, 105, 100, 98, 96, 94, 93, 95, 97, 99, 101, 103, 105]
    dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='h')
    df = pd.DataFrame({
        'open': prices,
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices
    }, index=dates)
    
    result = sim.run(df)
    
    print(f"Final Inventory: {result.final_inventory:.2f}")
    print(f"Realized PnL: ${result.realized_pnl:.2f}")
    print(f"Total Trades: {len(result.trades)}")
    
    for i, trade in enumerate(result.trades):
        print(f"Trade {i:2d}: {trade.side:4s} @ {trade.price:6.2f} (Grid Level: {trade.grid_level})")

    # Expectations:
    # - Should see 'buy' trades as it drops
    # - Should see 'sell' trades as it rises
    # - Grid levels should remap correctly (indices should stay within 0-10)
    for t in result.trades:
        assert 0 <= t.grid_level <= 10 or t.grid_level == -1, f"Invalid grid level {t.grid_level}"
    
    print("\n✅ Rolling Logic Verification Passed")

    
if __name__ == "__main__":
    test_rolling_grid_logic()
