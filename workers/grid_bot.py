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
from shared.config import load_config, get_redis_params

class GridBot:
    def __init__(self, symbol, grids, config_path='config/settings.yaml'):
        self.symbol = symbol
        self.grid_levels = grids
        self.worker_id = f"worker_{symbol.replace('/', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Load Config (with env substitution)
        self.config = load_config(config_path)

        # Setup Components
        self.bus = RedisBus(**get_redis_params(self.config))
        self.db = Database()
        self.order_manager = OrderManager(
            self.config['exchange']['name'],
            self.config['exchange']['api_key'],
            self.config['exchange']['secret'],
            testnet=(self.config['exchange']['mode'] == 'testnet')
        )
        
        self.logger = logging.getLogger(self.worker_id)
        logging.basicConfig(level=logging.INFO)
        
        # Load Strategy Params (Override CLI if exists)
        self.strategy_params = self._load_strategy_params()
        if self.strategy_params:
            self.grid_levels = self.strategy_params.get('grid_levels', self.grid_levels)
            self.logger.info(f"Loaded strategy for {self.symbol}: {self.strategy_params}")
        
        self.grids = [] # List of {'price': float, 'id': str, 'side': str}
        self.active_orders = []
        self.inventory = 0.0
        self.paused = False
        self.running = True

    def _load_strategy_params(self):
        try:
            strat_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategies.json')
            if os.path.exists(strat_path):
                with open(strat_path, 'r') as f:
                    strategies = json.load(f)
                    return strategies.get(self.symbol)
        except Exception as e:
            self.logger.error(f"Failed to load strategies: {e}")
        return None

    def calculate_grid_levels(self, current_price):
        if not self.strategy_params:
            self.logger.warning("No strategy params found. cannot calculate grids.")
            return

        lower = self.strategy_params['lower_limit']
        upper = self.strategy_params['upper_limit']
        
        if current_price < lower or current_price > upper:
            self.logger.warning(f"Price {current_price} is out of bounds [{lower}, {upper}].")
            return

        step = (upper - lower) / self.grid_levels
        
        self.grids = []
        for i in range(self.grid_levels + 1):
            price = lower + (i * step)
            self.grids.append({
                'price': price,
                'orders': [], # List of order_ids
            })
        self.logger.info(f"Calculated {len(self.grids)} grid levels.")

    def place_order(self, grid_index, side, amount):
        """Helper to place an order and track it in the grid."""
        grid = self.grids[grid_index]
        price = grid['price']
        try:
            if side == 'buy':
                order = self.order_manager.create_limit_buy(self.symbol, amount, price)
            else:
                order = self.order_manager.create_limit_sell(self.symbol, amount, price)
            
            grid['orders'].append({'id': order['id'], 'side': side})
            self.logger.info(f"Placed {side} order at {price} (Grid {grid_index})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to place {side} order at {price}: {e}")
            return False

    def place_initial_orders(self, current_price):
        if not self.grids:
            self.calculate_grid_levels(current_price)
            
        amount = self.strategy_params.get('amount_per_grid', 0.1)
        
        for i, grid in enumerate(self.grids):
            price = grid['price']
            # Skip if close to current price
            if abs(price - current_price) < (price * 0.005): 
                continue
                
            side = 'buy' if price < current_price else 'sell'
            self.place_order(i, side, amount)

    def check_orders(self):
        """Monitor open orders and handle fills."""
        try:
            # Get all open orders to check against
            open_orders = self.order_manager.fetch_open_orders(self.symbol)
            open_ids = set(o['id'] for o in open_orders)
            
            amount = self.strategy_params.get('amount_per_grid', 0.1)

            for i, grid in enumerate(self.grids):
                # Check all tracked orders in this grid
                # We iterate a copy to allow modification
                for order_data in list(grid['orders']):
                    order_id = order_data['id']
                    side = order_data['side']
                    
                    if order_id not in open_ids:
                        # Order is Missing -> Check Status
                        try:
                            # Verify if it was filled or canceled
                            fetched = self.order_manager.fetch_order(order_id, self.symbol)
                            status = fetched.get('status')
                            filled = fetched.get('filled', 0.0)
                        except Exception as e:
                            self.logger.error(f"Failed to fetch order {order_id}: {e}")
                            continue

                        if status == 'canceled':
                            self.logger.info(f"Order {order_id} was canceled. Removing.")
                            grid['orders'].remove(order_data)
                        
                        elif status == 'closed' or filled > 0:
                            self.logger.info(f"Order {order_id} ({side}) FILLED at level {i}.")
                            
                            # Log trade to SQLite
                            # TODO: self.db.log_order(fetched)

                            # Remove from tracking
                            grid['orders'].remove(order_data)
                            
                            # REBALANCING LOGIC (Ping Pong)
                            if side == 'buy':
                                self.inventory += filled # Track Inventory
                                
                                # Bought at i. Sell higher at i+1
                                target_idx = i + 1
                                if target_idx <= self.grid_levels: # Boundary Check
                                    self.logger.info(f"Rebalancing: Placing SELL at level {target_idx}")
                                    self.place_order(target_idx, 'sell', amount)
                                else:
                                    self.logger.warning("Price above grid! No more levels to sell.")
                            
                            elif side == 'sell':
                                self.inventory -= filled # Track Inventory
                                
                                # Sold at i. Buy lower at i-1
                                target_idx = i - 1
                                if target_idx >= 0: # Boundary Check
                                    self.logger.info(f"Rebalancing: Placing BUY at level {target_idx}")
                                    self.place_order(target_idx, 'buy', amount)
                                else:
                                    self.logger.warning("Price below grid! No more levels to buy.")
                            
        except Exception as e:
            self.logger.error(f"Error checking orders: {e}")

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

                if self.paused:
                    time.sleep(1)
                    continue

                # 2. Main Logic Tick (Mock)
                ticker = self.order_manager.fetch_ticker(self.symbol)
                last_price = ticker['last']
                
                # Logic: If no grids, initialize
                if not self.grids and self.strategy_params:
                     self.place_initial_orders(last_price)
                
                # Logic: Monitor orders
                self.check_orders()

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
                self.logger.info("Received PAUSE signal. Suspending operations.")
                self.paused = True
            elif cmd == 'RESUME':
                self.logger.info("Received RESUME signal. Resuming operations.")
                self.paused = False

    def report_status(self, current_price):
        # Update local DB
        self.db.update_worker_heartbeat(self.worker_id, self.symbol, 'RUNNING', 0.0)
        
        # Publish to Redis
        status_msg = {
            'worker_id': self.worker_id,
            'symbol': self.symbol,
            'status': 'RUNNING',
            'price': current_price,
            'inventory': self.inventory,
            'exposure': self.inventory * current_price,
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
