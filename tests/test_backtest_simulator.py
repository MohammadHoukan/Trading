import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.simulator import GridSimulator


def test_buy_fill_order_is_deterministic_and_path_consistent():
    # Two-candle setup:
    # - Candle 1 initializes around 20
    # - Candle 2 drops to 9, crossing both 15 and 10 buy levels
    # With only 16 capital and zero fees, exactly one 1-unit buy can fill.
    # Correct path-consistent order on a down move is 15 first, then 10.
    idx = pd.date_range('2025-01-01', periods=2, freq='h')
    df = pd.DataFrame(
        [
            {'open': 20.0, 'high': 20.0, 'low': 20.0, 'close': 20.0},
            {'open': 20.0, 'high': 20.0, 'low': 9.0, 'close': 9.0},
        ],
        index=idx,
    )

    sim = GridSimulator(
        lower_limit=10.0,
        upper_limit=30.0,
        grid_levels=4,
        amount_per_grid=1.0,
        initial_capital=16.0,
        fees_percent=0.0,
    )
    result = sim.run(df)

    assert len(result.trades) == 1
    assert result.trades[0].side == 'buy'
    assert result.trades[0].price == 15.0
    assert result.final_inventory == 1.0


def test_roll_grid_up_state_reindexes_levels_correctly():
    sim = GridSimulator(10.0, 30.0, 4, 1.0, rolling=True)
    buy_levels = {0, 2, 4}
    sell_levels = {0, 3}

    remapped_buys, remapped_sells, new_top_idx = sim._roll_grid_up_state(
        buy_levels, sell_levels
    )

    assert remapped_buys == {1, 3}
    assert remapped_sells == {2}
    assert new_top_idx == 4
    assert sim.grid_prices[0] == 15.0
    assert sim.grid_prices[-1] == 35.0


def test_roll_grid_down_state_reindexes_levels_correctly():
    sim = GridSimulator(10.0, 30.0, 4, 1.0, rolling=True)
    buy_levels = {0, 2, 4}
    sell_levels = {1, 4}

    remapped_buys, remapped_sells, new_bottom_idx = sim._roll_grid_down_state(
        buy_levels, sell_levels
    )

    assert remapped_buys == {1, 3}
    assert remapped_sells == {2}
    assert new_bottom_idx == 0
    assert sim.grid_prices[0] == 5.0
    assert sim.grid_prices[-1] == 25.0
