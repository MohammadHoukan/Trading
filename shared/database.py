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
