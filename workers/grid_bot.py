import argparse
import sys
import time
import logging
import json
import os
import uuid

# Add root directory to python path for imports to work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from workers.order_manager import OrderManager
from shared.messaging import RedisBus
from shared.database import Database
from shared.config import load_config

class GridBot:
    def __init__(self, symbol, grids, config_path='config/settings.yaml'):
        self.symbol = symbol
        self.grid_levels = grids
        self.worker_id = f"worker_{symbol.replace('/', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Load Config (with env substitution)
        self.config = load_config(config_path)

        # Setup Components
        redis_cfg = self.config['redis']
        self.bus = RedisBus(host=redis_cfg['host'], port=redis_cfg['port'], db=redis_cfg['db'])
        self.db = Database()
        self.order_manager = OrderManager(
            self.config['exchange']['name'],
            self.config['exchange']['api_key'],
            self.config['exchange']['secret'],
            testnet=(self.config['exchange']['mode'] == 'testnet')
        )
        
        self.logger = logging.getLogger(self.worker_id)
        logging.basicConfig(level=logging.INFO)
        
        self.running = True

    def run(self):
        self.logger.info(f"Starting Grid Bot for {self.symbol}")
        
        # Subscribe to commands
        pubsub = self.bus.subscribe(self.config['redis']['channels']['command'])
        
        while self.running:
            try:
                # 1. Check for messages
                msg = self.bus.get_message(pubsub)
                if msg:
                    self.handle_message(msg)

                # 2. Main Logic Tick (Mock)
                ticker = self.order_manager.fetch_ticker(self.symbol)
                last_price = ticker['last']
                
                # 3. Report Status
                self.report_status(last_price)
                
                time.sleep(2) # Throttle loop
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)

    def handle_message(self, msg):
        cmd = msg.get('command')
        target = msg.get('target')
        
        if target == 'all' or target == self.worker_id:
            if cmd == 'STOP':
                self.logger.warning("Received STOP signal.")
                self.running = False
            elif cmd == 'PAUSE':
                self.logger.info("Received PAUSE signal.")
                # Implement pause logic

    def report_status(self, current_price):
        # Update local DB
        self.db.update_worker_heartbeat(self.worker_id, self.symbol, 'RUNNING', 0.0)
        
        # Publish to Redis
        status_msg = {
            'worker_id': self.worker_id,
            'symbol': self.symbol,
            'status': 'RUNNING',
            'price': current_price,
            'timestamp': time.time()
        }
        self.bus.publish(self.config['redis']['channels']['status'], status_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True, help='Trading Pair (e.g. SOL/USDT)')
    parser.add_argument('--grids', type=int, default=20, help='Number of grid lines')
    args = parser.parse_args()

    bot = GridBot(args.pair, args.grids)
    bot.run()
