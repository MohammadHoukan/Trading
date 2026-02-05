"""
Grid trading simulation engine for backtesting.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: pd.Timestamp
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    grid_level: int
    pnl: float = 0.0  # For sells only


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    final_inventory: float
    final_avg_cost: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    initial_capital: float
    final_capital: float
    stop_loss_triggered: bool
    stop_loss_price: Optional[float]
    price_series: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)


class GridSimulator:
    """
    Simulates grid trading on historical price data.
    
    Grid Strategy Logic:
    - Place buy orders below current price at each grid level
    - Place sell orders above current price at each grid level
    - When a buy fills, place a sell one level higher
    - When a sell fills, place a buy one level lower
    """
    
    def __init__(
        self,
        lower_limit: float,
        upper_limit: float,
        grid_levels: int,
        amount_per_grid: float,
        initial_capital: float = 1000.0,
        stop_loss: Optional[float] = None,
        fees_percent: float = 0.1,  # 0.1% per trade (Binance default)
        rolling: bool = False  # Enable rolling/infinity grids
    ):
        """
        Initialize grid simulator.
        
        Args:
            lower_limit: Lowest grid price
            upper_limit: Highest grid price
            grid_levels: Number of grid lines
            amount_per_grid: Amount of base asset per grid order
            initial_capital: Starting capital in quote currency (e.g., USDT)
            stop_loss: Stop-loss price (optional)
            fees_percent: Trading fee percentage
            rolling: If True, grid shifts when price breaks bounds
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.grid_levels = grid_levels
        self.amount_per_grid = amount_per_grid
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.fees_percent = fees_percent
        self.rolling = rolling
        
        # Calculate grid prices
        self.grid_step = (upper_limit - lower_limit) / grid_levels
        self.grid_prices = [lower_limit + i * self.grid_step for i in range(grid_levels + 1)]
        
        mode_str = "ROLLING" if rolling else "FIXED"
        logger.info(f"Grid ({mode_str}): {grid_levels} levels from {lower_limit} to {upper_limit}, step={self.grid_step:.4f}")
    
    def run(self, ohlcv_df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on OHLCV data.
        
        Args:
            ohlcv_df: DataFrame with 'open', 'high', 'low', 'close' columns
            
        Returns:
            BacktestResult with all performance metrics
        """
        trades: List[Trade] = []
        
        # State
        inventory = 0.0
        avg_cost = 0.0
        realized_pnl = 0.0
        capital = self.initial_capital
        stop_loss_triggered = False
        stop_loss_price = None
        
        # Track which grid levels have pending orders
        # buy_levels: set of indices where we have buy orders
        # sell_levels: set of indices where we have sell orders
        buy_levels = set()
        sell_levels = set()
        
        # Equity curve for tracking
        equity_curve = []
        
        # Get first price to initialize grid
        first_price = ohlcv_df['close'].iloc[0]
        
        # Initialize grid orders
        for i, price in enumerate(self.grid_prices):
            if price < first_price * 0.995:  # Below current price = buy
                buy_levels.add(i)
            elif price > first_price * 1.005:  # Above current price = sell
                sell_levels.add(i)
        
        logger.info(f"Initialized {len(buy_levels)} buy levels, {len(sell_levels)} sell levels")
        
        # Simulate price movement
        for timestamp, row in ohlcv_df.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            
            # Check stop-loss
            if self.stop_loss and low <= self.stop_loss:
                stop_loss_triggered = True
                stop_loss_price = self.stop_loss
                
                # Close all positions at stop-loss price
                if inventory > 0:
                    pnl = (self.stop_loss - avg_cost) * inventory
                    fee = self.stop_loss * inventory * (self.fees_percent / 100)
                    realized_pnl += pnl - fee
                    capital += self.stop_loss * inventory - fee
                    
                    trades.append(Trade(
                        timestamp=timestamp,
                        side='sell',
                        price=self.stop_loss,
                        amount=inventory,
                        grid_level=-1,  # -1 indicates stop-loss
                        pnl=pnl - fee
                    ))
                    inventory = 0.0
                    avg_cost = 0.0
                
                logger.warning(f"Stop-loss triggered at {self.stop_loss}")
                break
            
            # Check for buy fills (price went down through grid levels)
            filled_buys = []
            for level in list(buy_levels):
                grid_price = self.grid_prices[level]
                if low <= grid_price:
                    # Buy filled!
                    cost = grid_price * self.amount_per_grid
                    fee = cost * (self.fees_percent / 100)
                    
                    if capital >= cost + fee:
                        capital -= cost + fee
                        
                        # Update inventory and avg cost
                        total_cost = (avg_cost * inventory) + (grid_price * self.amount_per_grid)
                        inventory += self.amount_per_grid
                        avg_cost = total_cost / inventory if inventory > 0 else 0.0
                        
                        trades.append(Trade(
                            timestamp=timestamp,
                            side='buy',
                            price=grid_price,
                            amount=self.amount_per_grid,
                            grid_level=level
                        ))
                        
                        filled_buys.append(level)
                        
                        # Place sell order one level higher
                        if level + 1 <= self.grid_levels:
                            sell_levels.add(level + 1)
                        elif self.rolling:
                            # Roll grid UP
                            self.grid_prices.pop(0)
                            new_top = self.grid_prices[-1] + self.grid_step
                            self.grid_prices.append(new_top)
                            sell_levels.add(len(self.grid_prices) - 1)
            
            for level in filled_buys:
                buy_levels.discard(level)
            
            # Check for sell fills (price went up through grid levels)
            filled_sells = []
            for level in list(sell_levels):
                grid_price = self.grid_prices[level]
                if high >= grid_price and inventory >= self.amount_per_grid:
                    # Sell filled!
                    amount = min(self.amount_per_grid, inventory)
                    proceeds = grid_price * amount
                    fee = proceeds * (self.fees_percent / 100)
                    
                    pnl = (grid_price - avg_cost) * amount - fee
                    realized_pnl += pnl
                    capital += proceeds - fee
                    inventory -= amount
                    
                    if inventory <= 0:
                        inventory = 0.0
                        avg_cost = 0.0
                    
                    trades.append(Trade(
                        timestamp=timestamp,
                        side='sell',
                        price=grid_price,
                        amount=amount,
                        grid_level=level,
                        pnl=pnl
                    ))
                    
                    filled_sells.append(level)
                    
                    # Place buy order one level lower
                    if level - 1 >= 0:
                        buy_levels.add(level - 1)
                    elif self.rolling:
                        # Roll grid DOWN
                        self.grid_prices.pop()
                        new_bottom = self.grid_prices[0] - self.grid_step
                        self.grid_prices.insert(0, new_bottom)
                        buy_levels.add(0)
                        # Adjust all level indices in buy_levels and sell_levels
                        buy_levels = {l + 1 for l in buy_levels if l >= 0}
                        sell_levels = {l + 1 for l in sell_levels}
            
            for level in filled_sells:
                sell_levels.discard(level)
            
            # Track equity
            unrealized = inventory * close - (avg_cost * inventory) if inventory > 0 else 0.0
            equity = capital + (inventory * close)
            equity_curve.append({'timestamp': timestamp, 'equity': equity, 'price': close})
        
        # Final calculations
        final_close = ohlcv_df['close'].iloc[-1]
        unrealized_pnl = (final_close - avg_cost) * inventory if inventory > 0 else 0.0
        total_pnl = realized_pnl + unrealized_pnl
        final_capital = capital + (inventory * final_close)
        
        # Build equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        return BacktestResult(
            trades=trades,
            final_inventory=inventory,
            final_avg_cost=avg_cost,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            stop_loss_triggered=stop_loss_triggered,
            stop_loss_price=stop_loss_price,
            price_series=ohlcv_df['close'],
            equity_curve=equity_df['equity'] if not equity_df.empty else pd.Series()
        )
