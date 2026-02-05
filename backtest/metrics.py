"""
Performance metrics calculation for backtests.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Backtest performance metrics."""
    total_pnl: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    num_round_trips: int
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    profit_factor: float


def calculate_metrics(
    trades: List,
    initial_capital: float,
    final_capital: float,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
) -> Metrics:
    """
    Calculate comprehensive backtest metrics.
    
    Args:
        trades: List of Trade objects
        initial_capital: Starting capital
        final_capital: Ending capital
        equity_curve: Series of equity values over time
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        
    Returns:
        Metrics dataclass with all performance stats
    """
    # Basic PnL
    total_pnl = final_capital - initial_capital
    return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    # Trade analysis
    sell_trades = [t for t in trades if t.side == 'sell']
    num_trades = len(trades)
    num_round_trips = len(sell_trades)
    
    if sell_trades:
        pnls = [t.pnl for t in sell_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        win_rate = (len(winners) / len(sell_trades)) * 100 if sell_trades else 0.0
        avg_trade_pnl = sum(pnls) / len(pnls) if pnls else 0.0
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0
        
        # Profit factor
        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 1.0  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        avg_trade_pnl = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        profit_factor = 0.0
    
    # Drawdown calculation
    max_drawdown_pct = 0.0
    if len(equity_curve) > 0:
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown_pct = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # Sharpe Ratio
    sharpe_ratio = 0.0
    if len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Annualize (assuming hourly data, ~8760 hours per year)
            periods_per_year = 8760
            excess_return = returns.mean() - (risk_free_rate / periods_per_year)
            sharpe_ratio = excess_return / returns.std() * np.sqrt(periods_per_year)
    
    return Metrics(
        total_pnl=total_pnl,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        num_trades=num_trades,
        num_round_trips=num_round_trips,
        avg_trade_pnl=avg_trade_pnl,
        best_trade=best_trade,
        worst_trade=worst_trade,
        profit_factor=profit_factor
    )


def format_metrics(metrics: Metrics) -> str:
    """Format metrics as a readable string."""
    return f"""
╔══════════════════════════════════════════════════╗
║           BACKTEST RESULTS                       ║
╠══════════════════════════════════════════════════╣
║  Total PnL:        ${metrics.total_pnl:>12,.2f}              ║
║  Return:           {metrics.return_pct:>12.2f}%              ║
║  Max Drawdown:     {metrics.max_drawdown_pct:>12.2f}%              ║
║  Sharpe Ratio:     {metrics.sharpe_ratio:>12.2f}               ║
╠══════════════════════════════════════════════════╣
║  Total Trades:     {metrics.num_trades:>12}               ║
║  Round Trips:      {metrics.num_round_trips:>12}               ║
║  Win Rate:         {metrics.win_rate:>12.1f}%              ║
║  Avg Trade PnL:    ${metrics.avg_trade_pnl:>12,.2f}              ║
║  Best Trade:       ${metrics.best_trade:>12,.2f}              ║
║  Worst Trade:      ${metrics.worst_trade:>12,.2f}              ║
║  Profit Factor:    {metrics.profit_factor:>12.2f}               ║
╚══════════════════════════════════════════════════╝
"""
