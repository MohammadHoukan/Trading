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
