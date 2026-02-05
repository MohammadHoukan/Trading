"""
CLI runner for grid trading backtests.

Usage:
    python -m backtest.runner --pair SOL/USDT --days 30 --grids 20
"""
import argparse
import sys
import os
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.data_fetcher import fetch_ohlcv
from backtest.simulator import GridSimulator
from backtest.execution_model import ExecutionModel
from backtest.metrics import calculate_metrics, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_strategy(symbol: str) -> dict | None:
    """Load strategy config from strategies.json."""
    strat_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategies.json')
    try:
        with open(strat_path, 'r') as f:
            strategies = json.load(f)
            return strategies.get(symbol)
    except Exception as e:
        logger.warning(f"Could not load strategy config: {e}")
        return None


def run_backtest(
    symbol: str,
    days: int,
    timeframe: str,
    lower_limit: float,
    upper_limit: float,
    grid_levels: int,
    amount_per_grid: float,
    initial_capital: float,
    stop_loss: float | None,
    save_trades: bool,
    rolling: bool = False,
    realistic: bool = False,
):
    """Run a single backtest."""
    print(f"\n📊 Fetching {days} days of {timeframe} data for {symbol}...")
    
    # Fetch data
    ohlcv = fetch_ohlcv(symbol, timeframe=timeframe, days=days)
    print(f"   Loaded {len(ohlcv)} candles from {ohlcv.index[0]} to {ohlcv.index[-1]}")
    
    # Create simulator
    print(f"\n⚙️  Grid Settings:")
    print(f"   Range: ${lower_limit:.2f} - ${upper_limit:.2f}")
    print(f"   Levels: {grid_levels}")
    print(f"   Amount/Grid: {amount_per_grid}")
    print(f"   Initial Capital: ${initial_capital:.2f}")
    if stop_loss:
        print(f"   Stop-Loss: ${stop_loss:.2f}")
    if rolling:
        print(f"   Mode: ROLLING (infinity grids)")
    if realistic:
        print(f"   Execution: REALISTIC (slippage/spread/partial fills)")

    execution_model = ExecutionModel(enabled=True) if realistic else None
    
    simulator = GridSimulator(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        grid_levels=grid_levels,
        amount_per_grid=amount_per_grid,
        initial_capital=initial_capital,
        stop_loss=stop_loss,
        rolling=rolling,
        execution_model=execution_model
    )
    
    # Run backtest
    print(f"\n🚀 Running backtest...")
    result = simulator.run(ohlcv)
    
    # Calculate metrics
    metrics = calculate_metrics(
        trades=result.trades,
        initial_capital=result.initial_capital,
        final_capital=result.final_capital,
        equity_curve=result.equity_curve
    )
    
    # Print results
    print(format_metrics(metrics))
    
    # Additional info
    print(f"\n📈 Position Summary:")
    print(f"   Final Inventory: {result.final_inventory:.4f}")
    print(f"   Avg Cost: ${result.final_avg_cost:.4f}")
    print(f"   Realized PnL: ${result.realized_pnl:.2f}")
    print(f"   Unrealized PnL: ${result.unrealized_pnl:.2f}")
    
    if result.stop_loss_triggered:
        print(f"\n⚠️  STOP-LOSS was triggered at ${result.stop_loss_price:.2f}")
    
    # Save trades if requested
    if save_trades and result.trades:
        trades_file = f"backtest_trades_{symbol.replace('/', '_')}.csv"
        import pandas as pd
        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'side': t.side,
                'price': t.price,
                'amount': t.amount,
                'grid_level': t.grid_level,
                'pnl': t.pnl
            }
            for t in result.trades
        ])
        trades_df.to_csv(trades_file, index=False)
        print(f"\n💾 Trades saved to {trades_file}")
    
    return result, metrics


def main():
    parser = argparse.ArgumentParser(description='Grid Trading Backtester')
    parser.add_argument('--pair', '-p', required=True, help='Trading pair (e.g., SOL/USDT)')
    parser.add_argument('--days', '-d', type=int, default=30, help='Days of history (default: 30)')
    parser.add_argument('--timeframe', '-t', default='1h', help='Candle timeframe (default: 1h)')
    parser.add_argument('--grids', '-g', type=int, help='Number of grid levels')
    parser.add_argument('--lower', '-l', type=float, help='Lower price limit')
    parser.add_argument('--upper', '-u', type=float, help='Upper price limit')
    parser.add_argument('--amount', '-a', type=float, help='Amount per grid')
    parser.add_argument('--capital', '-c', type=float, default=1000.0, help='Initial capital (default: 1000)')
    parser.add_argument('--stop-loss', '-s', type=float, help='Stop-loss price')
    parser.add_argument('--save-trades', action='store_true', help='Save trades to CSV')
    parser.add_argument('--rolling', '-r', action='store_true', help='Enable rolling/infinity grids')
    parser.add_argument('--realistic', action='store_true', help='Enable realistic execution model')
    
    args = parser.parse_args()
    
    # Try to load strategy config
    strategy = load_strategy(args.pair)
    
    # Use strategy config as defaults, CLI args override
    if strategy:
        logger.info(f"Loaded strategy config for {args.pair}")
        lower = args.lower if args.lower is not None else strategy.get('lower_limit')
        upper = args.upper if args.upper is not None else strategy.get('upper_limit')
        grids = args.grids if args.grids is not None else strategy.get('grid_levels', 20)
        amount = args.amount if args.amount is not None else strategy.get('amount_per_grid', 0.1)
        stop_loss = args.stop_loss if args.stop_loss is not None else strategy.get('stop_loss')
        rolling = args.rolling if args.rolling is not False else strategy.get('rolling_grids', False)
    else:
        lower = args.lower
        upper = args.upper
        grids = args.grids if args.grids is not None else 20
        amount = args.amount if args.amount is not None else 0.1
        stop_loss = args.stop_loss
        rolling = args.rolling
    
    # Validate required params
    if lower is None or upper is None:
        print("❌ Error: --lower and --upper are required (or configure in strategies.json)")
        sys.exit(1)
    
    run_backtest(
        symbol=args.pair,
        days=args.days,
        timeframe=args.timeframe,
        lower_limit=lower,
        upper_limit=upper,
        grid_levels=grids,
        amount_per_grid=amount,
        initial_capital=args.capital,
        stop_loss=stop_loss,
        save_trades=args.save_trades,
        rolling=rolling,
        realistic=args.realistic
    )


if __name__ == '__main__':
    main()
