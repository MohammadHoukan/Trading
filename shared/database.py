import sqlite3
import logging
from datetime import datetime

class Database:
    def __init__(self, db_path='swarm.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table for storing order history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            price REAL,
            amount REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Table for worker status/heartbeats
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS worker_status (
            worker_id TEXT PRIMARY KEY,
            symbol TEXT,
            status TEXT,
            last_heartbeat DATETIME,
            pnl REAL
        )
        ''')

        # Table for grid events (observability for ML training)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS grid_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            worker_id TEXT,
            symbol TEXT,
            event_type TEXT,
            side TEXT,
            price REAL,
            amount REAL,
            grid_level INTEGER,
            order_id TEXT,
            market_price REAL,
            grid_levels INTEGER,
            grid_spacing REAL,
            lower_limit REAL,
            upper_limit REAL,
            inventory REAL,
            avg_cost REAL,
            realized_profit REAL,
            source TEXT DEFAULT 'live'
        )
        ''')

        # Index for efficient querying by symbol and time
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_grid_events_symbol_time
        ON grid_events (symbol, timestamp)
        ''')

        conn.commit()
        conn.close()

    def get_connection(self):
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def log_order(self, order_data):
        """Log an order to the database."""
        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO orders (order_id, symbol, side, price, amount, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order_data['id'],
                    order_data['symbol'],
                    order_data['side'],
                    order_data['price'],
                    order_data['amount'],
                    order_data['status'],
                    datetime.now().isoformat()
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB Error log_order: {e}")

    def update_worker_heartbeat(self, worker_id, symbol, status, pnl):
        """Update worker heartbeat."""
        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO worker_status (worker_id, symbol, status, last_heartbeat, pnl)
                VALUES (?, ?, ?, ?, ?)
                ''', (worker_id, symbol, status, datetime.now().isoformat(), pnl))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB Error update_worker_heartbeat: {e}")

    def log_grid_event(self, event_data: dict):
        """
        Log a grid event for ML training data collection.

        event_data should contain:
            - timestamp: float (time.time())
            - worker_id: str
            - symbol: str
            - event_type: 'FILL' | 'PLACE' | 'CANCEL'
            - side: 'buy' | 'sell'
            - price: float
            - amount: float
            - grid_level: int
            - order_id: str (optional)
            - market_price: float (current market price at event time)
            - grid_levels: int (total grid levels)
            - grid_spacing: float (spacing between levels)
            - lower_limit: float
            - upper_limit: float
            - inventory: float (current inventory after event)
            - avg_cost: float (average cost basis)
            - realized_profit: float (total realized profit)
            - source: 'live' | 'backtest' (default: 'live')
        """
        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO grid_events (
                    timestamp, worker_id, symbol, event_type, side, price, amount,
                    grid_level, order_id, market_price, grid_levels, grid_spacing,
                    lower_limit, upper_limit, inventory, avg_cost, realized_profit, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_data.get('timestamp'),
                    event_data.get('worker_id'),
                    event_data.get('symbol'),
                    event_data.get('event_type'),
                    event_data.get('side'),
                    event_data.get('price'),
                    event_data.get('amount'),
                    event_data.get('grid_level'),
                    event_data.get('order_id'),
                    event_data.get('market_price'),
                    event_data.get('grid_levels'),
                    event_data.get('grid_spacing'),
                    event_data.get('lower_limit'),
                    event_data.get('upper_limit'),
                    event_data.get('inventory'),
                    event_data.get('avg_cost'),
                    event_data.get('realized_profit'),
                    event_data.get('source', 'live'),
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB Error log_grid_event: {e}")

    # ========== ML Training Data Query Methods ==========

    def get_grid_events(self, symbol: str = None, since: float = None,
                       source: str = None, event_type: str = None) -> list:
        """
        Query grid events for ML training.

        Args:
            symbol: Filter by trading pair (optional)
            since: Filter events after this timestamp (optional)
            source: Filter by source ('live' or 'backtest', optional)
            event_type: Filter by event type ('FILL', 'PLACE', 'CANCEL', optional)

        Returns:
            List of event dicts
        """
        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()

                query = 'SELECT * FROM grid_events WHERE 1=1'
                params = []

                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                if since:
                    query += ' AND timestamp > ?'
                    params.append(since)
                if source:
                    query += ' AND source = ?'
                    params.append(source)
                if event_type:
                    query += ' AND event_type = ?'
                    params.append(event_type)

                query += ' ORDER BY timestamp ASC'

                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                return [dict(zip(columns, row)) for row in rows]
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB Error get_grid_events: {e}")
            return []

    def get_backtest_metrics(self, symbol: str, lookback_days: int = 30) -> dict:
        """
        Calculate backtest metrics for a symbol from grid_events.

        Args:
            symbol: Trading pair
            lookback_days: Number of days to look back

        Returns:
            Dict with metrics: sharpe_ratio, win_rate, max_drawdown, profit_factor,
                              avg_trade_return, trade_count
        """
        import time
        since = time.time() - (lookback_days * 24 * 3600)

        events = self.get_grid_events(
            symbol=symbol,
            since=since,
            event_type='FILL'
        )

        if not events:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.5,
                'max_drawdown': 0.0,
                'profit_factor': 1.0,
                'avg_trade_return': 0.0,
                'trade_count': 0,
            }

        # Calculate metrics
        trades = []
        buy_queue = []

        for event in events:
            side = event.get('side')
            price = event.get('price', 0)
            amount = event.get('amount', 0)

            if side == 'buy':
                buy_queue.append({'price': price, 'amount': amount})
            elif side == 'sell' and buy_queue:
                # FIFO matching
                buy = buy_queue.pop(0)
                trade_return = (price - buy['price']) / buy['price']
                trades.append(trade_return)

        if not trades:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.5,
                'max_drawdown': 0.0,
                'profit_factor': 1.0,
                'avg_trade_return': 0.0,
                'trade_count': 0,
            }

        import numpy as np
        trades = np.array(trades)

        # Win rate
        win_rate = np.sum(trades > 0) / len(trades) if len(trades) > 0 else 0.5

        # Sharpe ratio (simplified - annualized assuming hourly trades)
        mean_return = np.mean(trades)
        std_return = np.std(trades) if len(trades) > 1 else 1.0
        sharpe_ratio = (mean_return / std_return) * np.sqrt(8760) if std_return > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(trades[trades > 0]) if np.any(trades > 0) else 0.0
        gross_loss = abs(np.sum(trades[trades < 0])) if np.any(trades < 0) else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else min(gross_profit * 10, 100.0)

        # Max drawdown (simplified from cumulative returns)
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        return {
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'profit_factor': min(float(profit_factor), 100.0),
            'avg_trade_return': float(mean_return),
            'trade_count': len(trades),
        }

    def get_fill_statistics(self, symbol: str, lookback_hours: int = 24) -> dict:
        """
        Get fill rate statistics for a symbol.

        Args:
            symbol: Trading pair
            lookback_hours: Hours to look back

        Returns:
            Dict with fill_rate, total_fills, total_places
        """
        import time
        since = time.time() - (lookback_hours * 3600)

        try:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        SUM(CASE WHEN event_type = 'FILL' THEN 1 ELSE 0 END) as fills,
                        SUM(CASE WHEN event_type = 'PLACE' THEN 1 ELSE 0 END) as places
                    FROM grid_events
                    WHERE symbol = ? AND timestamp > ?
                ''', (symbol, since))

                row = cursor.fetchone()

                if row and row[1] and row[1] > 0:
                    return {
                        'fill_rate': row[0] / row[1],
                        'total_fills': row[0],
                        'total_places': row[1],
                    }
                return {
                    'fill_rate': None,
                    'total_fills': 0,
                    'total_places': 0,
                }
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"DB Error get_fill_statistics: {e}")
            return {'fill_rate': None, 'total_fills': 0, 'total_places': 0}
