
import unittest
import pandas as pd
from backtest.simulator import GridSimulator

class TestSimulatorPhases(unittest.TestCase):
    def test_green_candle_phases(self):
        """
        Green Candle (Open < Close): Should Buy (Low) then Sell (High).
        If we start with 0 inventory:
        1. Price drops to Low -> BUY triggers -> Inventory +1
        2. Price rises to High -> SELL triggers -> Inventory -1
        Result: Inventory should be 0 (Round trip complete).
        """
        # Grid setup: Buy at 95, Sell at 105
        # Range 90-110, 2 grids => 90, 100, 110. (Wait, 2 grids means 3 lines?)
        # Let's simple it: Lower 90, Upper 110, Grids 2.
        # Step = 10. Lines at 90, 100, 110.
        # Buy level roughly below 100.
        
        sim = GridSimulator(
            lower_limit=90.0,
            upper_limit=110.0,
            grid_levels=2, # Lines at 90, 100, 110
            amount_per_grid=1.0,
            initial_capital=1000.0
        )
        
        # Green Candle: Open 100, Low 90, High 110, Close 110
        # Expected Flow:
        # 1. Start Price 100.
        # 2. Phase 1 (Buy): Low is 90. Grid at 90 is triggered (<= 90). Buy fill.
        #    Inv -> 1.0. Sell order placed at 100 (one level up? No, grid step is 10).
        #    If buy at 90 (index 0). Sell placed at 100 (index 1).
        # 3. Phase 2 (Sell): High is 110. Sell order at 100 is triggered (>= 100).
        #    Inv -> 0.0.
        
        df = pd.DataFrame([{
            'open': 100.0,
            'high': 110.0,
            'low': 90.0,
            'close': 110.0,
            'volume': 1000.0
        }], index=[pd.Timestamp.now()])
        
        result = sim.run(df)
        
        # Verify Inventory is 0 (Round trip happened)
        self.assertEqual(result.final_inventory, 0.0, "Green candle should complete round trip (Buy then Sell)")
        self.assertEqual(len(result.trades), 2, "Should have 2 trades (Buy + Sell)")

    def test_red_candle_phases(self):
        """
        Red Candle (Open > Close): Should Sell (High) then Buy (Low).
        If we start with 0 inventory:
        1. Price rises to High -> SELL triggers -> Fails (0 Inventory).
        2. Price drops to Low -> BUY triggers -> Inventory +1.
        Result: Inventory should be 1 (Only Buy happened).
        
        If phases were wrong (Green logic on Red candle):
        1. Buy triggers first (at Low) -> Inv +1.
        2. Sell triggers second (at High) -> Inv -1.
        Result would be 0. We expect 1.
        """
        sim = GridSimulator(
            lower_limit=90.0,
            upper_limit=110.0,
            grid_levels=2,
            amount_per_grid=1.0,
            initial_capital=1000.0
        )
        
        # Red Candle: Open 100, High 110, Low 90, Close 90
        df = pd.DataFrame([{
            'open': 100.0,
            'high': 110.0,
            'low': 90.0,
            'close': 90.0,
            'volume': 1000.0
        }], index=[pd.Timestamp.now()])
        
        result = sim.run(df)
        
        # Verify Inventory is 1 (Only Buy happened, Sell failed due to ordering)
        self.assertEqual(result.final_inventory, 1.0, "Red candle should only Buy (Sell phase happens before Buy phase, so no inventory to sell)")
        self.assertEqual(len(result.trades), 1, "Should have 1 trade (Buy only)")

if __name__ == '__main__':
    unittest.main()
