import pandas as pd
from backtest.simulator import GridSimulator
import logging

logging.basicConfig(level=logging.DEBUG)

def reproduce_green():
    print("\n--- REPRODUCING GREEN CANDLE ---")
    sim = GridSimulator(
        lower_limit=90.0,
        upper_limit=110.0,
        grid_levels=2,
        amount_per_grid=1.0,
        initial_capital=1000.0,
        symbol="SOL/USDT"
    )
    # Open 100, Low 90, High 110, Close 110
    df = pd.DataFrame([{
        'open': 100.0, 'high': 110.0, 'low': 90.0, 'close': 110.0, 'volume': 1000.0
    }], index=[pd.Timestamp.now()])
    
    result = sim.run(df)
    print(f"Green Result: Trades={len(result.trades)}, Inv={result.final_inventory}")
    for t in result.trades:
        print(f"  Trade: {t.side} @ {t.price} (Grid {t.grid_level})")

def reproduce_red():
    print("\n--- REPRODUCING RED CANDLE ---")
    sim = GridSimulator(
        lower_limit=90.0,
        upper_limit=110.0,
        grid_levels=2,
        amount_per_grid=1.0,
        initial_capital=1000.0,
        symbol="SOL/USDT"
    )
    # Open 100, High 110, Low 90, Close 90
    df = pd.DataFrame([{
        'open': 100.0, 'high': 110.0, 'low': 90.0, 'close': 90.0, 'volume': 1000.0
    }], index=[pd.Timestamp.now()])
    
    result = sim.run(df)
    print(f"Red Result: Trades={len(result.trades)}, Inv={result.final_inventory}")
    for t in result.trades:
        print(f"  Trade: {t.side} @ {t.price} (Grid {t.grid_level})")

if __name__ == "__main__":
    reproduce_green()
    reproduce_red()
