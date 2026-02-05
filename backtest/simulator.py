"""
Grid trading simulation engine for backtesting.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

try:
    from backtest.execution_model import ExecutionModel
except ImportError:
    ExecutionModel = None

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
        rolling: bool = False,  # Enable rolling/infinity grids
        trend_filter_period: Optional[int] = None,  # SMA period for trend filter (e.g., 50)
        database = None,  # Optional: Shared Database instance for logging
        symbol: str = "SOL/USDT", # Symbol for logging
        execution_model = None  # Optional: ExecutionModel for realistic fills
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
            database: Optional Database instance for logging events
            symbol: Symbol for logging
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.grid_levels = grid_levels
        self.amount_per_grid = amount_per_grid
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.fees_percent = fees_percent
        self.rolling = rolling
        self.database = database
        self.symbol = symbol
        self.execution_model = execution_model
        
        # Calculate grid prices
        self.grid_step = (upper_limit - lower_limit) / grid_levels
        self.grid_prices = [lower_limit + i * self.grid_step for i in range(grid_levels + 1)]
        
        mode_str = "ROLLING" if rolling else "FIXED"
        logger.info(f"Grid ({mode_str}): {grid_levels} levels from {lower_limit} to {upper_limit}, step={self.grid_step:.4f}")
        
        self.trend_filter_period = trend_filter_period
        if trend_filter_period:
            logger.info(f"Trend filter enabled: SMA-{trend_filter_period}")

    def _log_event(self, event_type, side, price, amount, grid_level, market_price, timestamp,
                   inventory=0.0, avg_cost=0.0, realized_profit=0.0):
        """Helper to log event to database if configured."""
        if self.database:
            event_data = {
                'timestamp': timestamp.timestamp(),
                'worker_id': 'backtest_simulator',
                'symbol': self.symbol,
                'event_type': event_type,
                'side': side,
                'price': price,
                'amount': amount,
                'grid_level': grid_level,
                'order_id': f"sim_{event_type}_{grid_level}_{timestamp.timestamp()}",
                'market_price': market_price,
                'grid_levels': self.grid_levels,
                'grid_spacing': self.grid_step,
                'lower_limit': self.lower_limit,
                'upper_limit': self.upper_limit,
                'inventory': inventory,
                'avg_cost': avg_cost,
                'realized_profit': realized_profit,
                'source': 'backtest',
            }
            self.database.log_grid_event(event_data)

    def _roll_grid_up_state(self, buy_levels: set, sell_levels: set):
        """
        Shift grid one step up and remap pending level indices.

        Old index 0 is removed; all remaining levels shift down by 1.
        """
        self.grid_prices.pop(0)
        new_top = self.grid_prices[-1] + self.grid_step
        self.grid_prices.append(new_top)

        remapped_buys = {level - 1 for level in buy_levels if level > 0}
        remapped_sells = {level - 1 for level in sell_levels if level > 0}
        return remapped_buys, remapped_sells, len(self.grid_prices) - 1

    def _roll_grid_down_state(self, buy_levels: set, sell_levels: set):
        """
        Shift grid one step down and remap pending level indices.

        Old top index is removed; all remaining levels shift up by 1.
        """
        old_top_index = len(self.grid_prices) - 1
        self.grid_prices.pop()
        new_bottom = self.grid_prices[0] - self.grid_step
        self.grid_prices.insert(0, new_bottom)

        remapped_buys = {level + 1 for level in buy_levels if level < old_top_index}
        remapped_sells = {level + 1 for level in sell_levels if level < old_top_index}
        return remapped_buys, remapped_sells, 0
    
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
                self._log_event('PLACE', 'buy', price, self.amount_per_grid, i, first_price, ohlcv_df.index[0],
                               inventory, avg_cost, realized_pnl)
            elif price > first_price * 1.005:  # Above current price = sell
                sell_levels.add(i)
                self._log_event('PLACE', 'sell', price, self.amount_per_grid, i, first_price, ohlcv_df.index[0],
                               inventory, avg_cost, realized_pnl)
        
        logger.info(f"Initialized {len(buy_levels)} buy levels, {len(sell_levels)} sell levels")
        
        # Calculate trend filter (SMA)
        sma = None
        if self.trend_filter_period:
            sma = ohlcv_df['close'].rolling(window=self.trend_filter_period).mean()
        
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
            
            # Determine intra-candle execution order
            # Green (Open <= Close): Low -> High (Buy then Sell)
            # Red   (Open > Close):  High -> Low (Sell then Buy)
            phases = ['buy', 'sell'] if row['open'] <= row['close'] else ['sell', 'buy']
            print(f"DEBUG: Row Open={row['open']} Close={row['close']} Phases={phases}")
            
            for phase in phases:
                print(f"  DEBUG: Starting Phase {phase}. Inv={inventory}. Buys={buy_levels}, Sells={sell_levels}")
                if phase == 'buy':
                    # Trend Filter: Skip buying if price is below SMA (downtrend)
                    if sma is not None:
                        current_sma = sma.get(timestamp)
                        if current_sma is not None and not pd.isna(current_sma):
                            if close < current_sma:
                                continue  # Skip buying in downtrend
                    
                    # Check for buy fills (price went down through grid levels)
                    for level in sorted(buy_levels, reverse=True):
                        if level not in buy_levels or level >= len(self.grid_prices):
                            continue
                        grid_price = self.grid_prices[level]
                        if low <= grid_price:
                            # Simulate realistic fill if model is enabled
                            fill_amount = self.amount_per_grid
                            fill_price = grid_price
                            was_filled = True
                            
                            if self.execution_model:
                                fill_amount, fill_price, was_filled = self.execution_model.simulate_fill(
                                    'buy', grid_price, self.amount_per_grid, close
                                )
                            
                            if not was_filled or fill_amount <= 0:
                                continue

                            # Buy filled!
                            cost = fill_price * fill_amount
                            fee = cost * (self.fees_percent / 100)
                            
                            if capital >= cost + fee:
                                capital -= cost + fee
                                
                                # Update inventory and avg cost
                                total_cost = (avg_cost * inventory) + cost + fee
                                inventory += fill_amount
                                avg_cost = total_cost / inventory if inventory > 0 else 0.0
                                
                                trades.append(Trade(
                                    timestamp=timestamp,
                                    side='buy',
                                    price=fill_price,
                                    amount=fill_amount,
                                    grid_level=level
                                ))
                                buy_levels.discard(level)

                                self._log_event('FILL', 'buy', fill_price, fill_amount, level, low, timestamp,
                                               inventory, avg_cost, realized_pnl)

                                # Place sell order one level higher
                                if level + 1 <= self.grid_levels:
                                    sell_levels.add(level + 1)
                                    self._log_event('PLACE', 'sell', self.grid_prices[level+1], self.amount_per_grid, level+1, low, timestamp,
                                                   inventory, avg_cost, realized_pnl)
                                elif self.rolling:
                                    buy_levels, sell_levels, new_top_idx = self._roll_grid_up_state(
                                        buy_levels, sell_levels
                                    )
                                    sell_levels.add(new_top_idx)

                elif phase == 'sell':
                    # Check for sell fills (price went up through grid levels)
                    for level in sorted(sell_levels):
                        if level not in sell_levels or level >= len(self.grid_prices):
                            continue
                        grid_price = self.grid_prices[level]
                        if high >= grid_price and inventory > 0:
                            # Simulate realistic fill if model is enabled
                            order_amount = min(self.amount_per_grid, inventory)
                            fill_amount = order_amount
                            fill_price = grid_price
                            was_filled = True
                            
                            if self.execution_model:
                                fill_amount, fill_price, was_filled = self.execution_model.simulate_fill(
                                    'sell', grid_price, order_amount, close
                                )
                            
                            if not was_filled or fill_amount <= 0:
                                continue

                            # Sell filled!
                            proceeds = fill_price * fill_amount
                            fee = proceeds * (self.fees_percent / 100)
                            
                            pnl = (fill_price - avg_cost) * fill_amount - fee
                            realized_pnl += pnl
                            capital += proceeds - fee
                            inventory -= fill_amount
                            
                            if inventory <= 0:
                                inventory = 0.0
                                avg_cost = 0.0
                            
                            trades.append(Trade(
                                timestamp=timestamp,
                                side='sell',
                                price=fill_price,
                                amount=fill_amount,
                                grid_level=level,
                                pnl=pnl
                            ))
                            sell_levels.discard(level)

                            self._log_event('FILL', 'sell', fill_price, fill_amount, level, high, timestamp,
                                           inventory, avg_cost, realized_pnl)

                            # Place buy order one level lower
                            if level - 1 >= 0:
                                buy_levels.add(level - 1)
                                self._log_event('PLACE', 'buy', self.grid_prices[level-1], self.amount_per_grid, level-1, high, timestamp,
                                               inventory, avg_cost, realized_pnl)
                            elif self.rolling:
                                buy_levels, sell_levels, new_bottom_idx = self._roll_grid_down_state(
                                    buy_levels, sell_levels
                                )
                                buy_levels.add(new_bottom_idx)
            
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
