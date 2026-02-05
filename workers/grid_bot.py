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
from shared.rate_limiter import RateLimiter
import hashlib
import threading

# Constants
STALE_DATA_THRESHOLD_SECONDS = 15
COMMAND_STREAM = 'swarm:commands'
COMMAND_GROUP = 'workers'

class GridBot:
    def __init__(self, symbol, grids, config_path='config/settings.yaml'):
        self.symbol = symbol
        self.grid_levels = grids
        self.worker_id = f"worker_{symbol.replace('/', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Load Config (with env substitution)
        self.config = load_config(config_path)
        self.logger = logging.getLogger(self.worker_id)
        logging.basicConfig(level=logging.INFO)
        self.running = True

        # Setup Components
        # Setup Components
        self.bus = RedisBus(**get_redis_params(self.config))
        self.db = Database()
        
        # KEY POOL LOCKING LOGIC
        self.key_lock_id = None
        self.api_key = None
        self.secret = None
        
        if 'pool' in self.config['exchange']:
            self.logger.info("Attempting to acquire API Key from Pool...")
            acquired = False
            for creds in self.config['exchange']['pool']:
                k = creds['api_key']
                s = creds['secret']
                
                # Check if placeholder
                if k.startswith("${") or not k:
                    continue
                    
                key_hash = hashlib.md5(k.encode()).hexdigest()
                lock_key = f"lock:apikey:{key_hash}"
                
                # Try to acquire lock (TTL 60s)
                # We use the RedisBus connection for this raw command
                if self.bus.set(lock_key, self.worker_id, nx=True, ex=60):
                    self.logger.info(f"Acquired Lock for Key: ...{k[-4:]}")
                    self.api_key = k
                    self.secret = s
                    self.key_lock_id = lock_key
                    acquired = True
                    break
            
            if not acquired:
                self.logger.critical("NO AVAILABLE API KEYS IN POOL! All Locked.")
                raise RuntimeError("Key Pool Exhausted")
                
            # Start Heartbeat Thread for Lock Renewal
            self.lock_thread = threading.Thread(target=self._renew_lock_loop, daemon=True)
            self.lock_thread.start()
            
        else:
            # Fallback to legacy single key
            self.logger.warning("Using Single Default Key (Race Conditions Possible!)")
            self.api_key = self.config['exchange']['api_key']
            self.secret = self.config['exchange']['secret']

        self.order_manager = OrderManager(
            self.config['exchange']['name'],
            self.api_key,
            self.secret,
            testnet=(self.config['exchange']['mode'] == 'testnet')
        )
        
        # Load Strategy Params (Override CLI if exists)
        self.strategy_params = self._load_strategy_params()
        if not self.strategy_params:
            self.logger.error(
                f"No strategy found for {self.symbol}. Add it to config/strategies.json."
            )
            raise ValueError(f"Missing strategy for {self.symbol}")

        if not self.strategy_params.get('enabled', True):
            self.logger.error(f"Strategy for {self.symbol} is disabled.")
            raise ValueError(f"Strategy disabled for {self.symbol}")

        self.grid_levels = self.strategy_params.get('grid_levels', self.grid_levels)
        self._validate_strategy_params()
        self.logger.info(f"Loaded strategy for {self.symbol}: {self.strategy_params}")
        
        self.grids = []  # List of {'price': float, 'orders': [{'id', 'side', 'amount', 'price'}]}
        self.active_orders = set()
        self.inventory = 0.0
        self.avg_cost = 0.0
        self.realized_profit = 0.0
        self.paused = False
        
        # Production safety features
        self.rate_limiter = RateLimiter(self.bus, max_requests=8, window_seconds=1)
        self.last_price_update = time.time()
        self.stop_loss_triggered = False
        
        # Setup Redis Streams consumer group for reliable commands
        self.bus.create_consumer_group(COMMAND_STREAM, COMMAND_GROUP, start_id='$')
        
        # Rolling grids (infinity grids) settings
        self.rolling_grids_enabled = self.strategy_params.get('rolling_grids', False)
        self.grid_step = 0.0  # Calculated when grid is initialized
        if self.rolling_grids_enabled:
            self.logger.info("Rolling Grids ENABLED - grid will shift with price")

    def _renew_lock_loop(self):
        """Background thread to keep the API Key lock alive."""
        while self.running:
            try:
                if self.key_lock_id and self.worker_id:
                    # Lua script to verify ownership before extending
                    script = """
                    if redis.call("get", KEYS[1]) == ARGV[1] then
                        return redis.call("expire", KEYS[1], ARGV[2])
                    else
                        return 0
                    end
                    """
                    result = self.bus.r.eval(script, 1, self.key_lock_id, self.worker_id, 60)
                    if not result:
                        self.logger.warning("Lost API Key lock ownership! Stopping bot.")
                        self.key_lock_id = None
                        self.running = False  # Fix #6: Stop bot on lock loss
                        self._publish_terminal_status('ERROR')
            except Exception as e:
                self.logger.error(f"Failed to renew API Key lock: {e}")
            time.sleep(30)

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

    def _validate_strategy_params(self):
        required = ['lower_limit', 'upper_limit']
        for key in required:
            if key not in self.strategy_params:
                raise ValueError(f"Strategy missing required key: {key}")

        if not isinstance(self.grid_levels, int) or self.grid_levels <= 0:
            raise ValueError("grid_levels must be a positive integer")

        lower = self.strategy_params['lower_limit']
        upper = self.strategy_params['upper_limit']
        if lower >= upper:
            raise ValueError("lower_limit must be less than upper_limit")

        amount = self.strategy_params.get('amount_per_grid', None)
        if amount is not None and amount <= 0:
            raise ValueError("amount_per_grid must be > 0")

    def _log_grid_event(self, event_type: str, side: str, price: float, amount: float,
                        grid_level: int, order_id: str = None, market_price: float = None):
        """
        Log a grid event for ML training data collection.

        Args:
            event_type: 'FILL', 'PLACE', or 'CANCEL'
            side: 'buy' or 'sell'
            price: order price
            amount: order amount
            grid_level: index of the grid level
            order_id: exchange order ID (optional)
            market_price: current market price (optional, will fetch if not provided)
        """
        try:
            if market_price is None:
                ticker = self.order_manager.fetch_ticker(self.symbol)
                market_price = ticker.get('last', price) if ticker else price

            event_data = {
                'timestamp': time.time(),
                'worker_id': self.worker_id,
                'symbol': self.symbol,
                'event_type': event_type,
                'side': side,
                'price': price,
                'amount': amount,
                'grid_level': grid_level,
                'order_id': order_id,
                'market_price': market_price,
                'grid_levels': self.grid_levels,
                'grid_spacing': self.grid_step,
                'lower_limit': self.strategy_params.get('lower_limit'),
                'upper_limit': self.strategy_params.get('upper_limit'),
                'inventory': self.inventory,
                'avg_cost': self.avg_cost,
                'realized_profit': self.realized_profit,
            }
            self.db.log_grid_event(event_data)
        except Exception as e:
            self.logger.warning(f"Failed to log grid event: {e}")

    def _has_existing_order(self, grid_index, side):
        grid = self.grids[grid_index]
        return any(order['side'] == side for order in grid['orders'])

    def _roll_grid_up(self, amount):
        """
        Roll the grid UP when price breaks the top boundary.
        - Cancel lowest buy order (now far away)
        - Shift all grid levels up by one step
        - Place new sell order at the new top
        """
        self.logger.info("📈 ROLLING GRID UP - Price broke top boundary")
        
        # Cancel the lowest buy order (grid index 0)
        if self.grids and self.grids[0]['orders']:
            for order_data in list(self.grids[0]['orders']):
                if order_data['side'] == 'buy':
                    try:
                        self.order_manager.cancel_order(order_data['id'], self.symbol)
                        self.logger.info(f"Cancelled bottom buy order at {order_data['price']:.4f}")
                    except Exception as e:
                        self.logger.error(f"Failed to cancel order: {e}")
                    self.grids[0]['orders'].remove(order_data)
                    self.active_orders.discard(order_data['id'])
        
        # Shift grid: remove bottom level, add new top level
        old_top = self.grids[-1]['price']
        new_top_price = old_top + self.grid_step
        
        self.grids.pop(0)  # Remove bottom
        self.grids.append({
            'price': new_top_price,
            'orders': []
        })
        
        # Place new sell order at the new top
        self.logger.info(f"Grid shifted up. New range: [{self.grids[0]['price']:.4f}, {new_top_price:.4f}]")
        self.place_order(len(self.grids) - 1, 'sell', amount)

    def _roll_grid_down(self, amount):
        """
        Roll the grid DOWN when price breaks the bottom boundary.
        - Cancel highest sell order (now far away)
        - Shift all grid levels down by one step
        - Place new buy order at the new bottom
        """
        self.logger.info("📉 ROLLING GRID DOWN - Price broke bottom boundary")
        
        # Cancel the highest sell order (last grid index)
        if self.grids and self.grids[-1]['orders']:
            for order_data in list(self.grids[-1]['orders']):
                if order_data['side'] == 'sell':
                    try:
                        self.order_manager.cancel_order(order_data['id'], self.symbol)
                        self.logger.info(f"Cancelled top sell order at {order_data['price']:.4f}")
                    except Exception as e:
                        self.logger.error(f"Failed to cancel order: {e}")
                    self.grids[-1]['orders'].remove(order_data)
                    self.active_orders.discard(order_data['id'])
        
        # Shift grid: remove top level, add new bottom level
        old_bottom = self.grids[0]['price']
        new_bottom_price = old_bottom - self.grid_step
        
        self.grids.pop()  # Remove top
        self.grids.insert(0, {
            'price': new_bottom_price,
            'orders': []
        })
        
        # Place new buy order at the new bottom
        self.logger.info(f"Grid shifted down. New range: [{new_bottom_price:.4f}, {self.grids[-1]['price']:.4f}]")
        self.place_order(0, 'buy', amount)

    def cancel_open_orders(self):
        try:
            open_orders = self.order_manager.fetch_open_orders(self.symbol)
            for order in open_orders:
                order_id = order.get('id')
                if not order_id:
                    continue
                try:
                    self.order_manager.cancel_order(order_id, self.symbol)
                    # Log CANCEL event for ML training data
                    # Look up grid level from tracked orders
                    grid_level = -1
                    side = order.get('side', 'unknown')
                    price = order.get('price', 0.0)
                    amount = order.get('amount', 0.0)
                    for i, grid in enumerate(self.grids):
                        for tracked in grid['orders']:
                            if tracked['id'] == order_id:
                                grid_level = i
                                side = tracked.get('side', side)
                                price = tracked.get('price', price)
                                amount = tracked.get('amount', amount)
                                break
                        if grid_level >= 0:
                            break
                    self._log_grid_event(
                        event_type='CANCEL',
                        side=side,
                        price=price,
                        amount=amount,
                        grid_level=grid_level,
                        order_id=order_id
                    )
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order_id}: {e}")

            # Reconcile cancel/fill outcomes and keep unresolved orders tracked.
            self.check_orders(allow_rebalance=False)
        except Exception as e:
            self.logger.error(f"Failed to cancel open orders: {e}")

    def _apply_fill(self, side, amount, price):
        if not amount or amount <= 0:
            return
        if side == 'buy':
            total_cost = (self.avg_cost * self.inventory) + (price * amount)
            self.inventory += amount
            self.avg_cost = total_cost / self.inventory if self.inventory > 0 else 0.0
        elif side == 'sell':
            if self.inventory <= 0:
                self.logger.warning("Sell fill with zero inventory; ignoring for PnL.")
                return
            if amount > self.inventory:
                self.logger.warning(
                    f"Sell fill exceeds inventory ({amount} > {self.inventory}). "
                    "Clamping to available inventory."
                )
                amount = self.inventory
            self.realized_profit += (price - self.avg_cost) * amount
            self.inventory -= amount
            if self.inventory <= 0:
                self.inventory = 0.0
                self.avg_cost = 0.0

    def _rebalance_after_fill(self, grid_index, side, amount, allow_rebalance=True):
        if not allow_rebalance:
            return
        if side == 'buy':
            # Bought at i. Sell higher at i+1
            target_idx = grid_index + 1
            if target_idx <= self.grid_levels:  # Boundary Check
                self.logger.info(f"Rebalancing: Placing SELL at level {target_idx}")
                self.place_order(target_idx, 'sell', amount)
            elif self.rolling_grids_enabled:
                # At top boundary - roll grid up
                self._roll_grid_up(amount)
            else:
                self.logger.warning("Price above grid! No more levels to sell.")
        elif side == 'sell':
            # Sold at i. Buy lower at i-1
            target_idx = grid_index - 1
            if target_idx >= 0:  # Boundary Check
                self.logger.info(f"Rebalancing: Placing BUY at level {target_idx}")
                self.place_order(target_idx, 'buy', amount)
            elif self.rolling_grids_enabled:
                # At bottom boundary - roll grid down
                self._roll_grid_down(amount)
            else:
                self.logger.warning("Price below grid! No more levels to buy.")

    def calculate_grid_levels(self, current_price):
        if not self.strategy_params:
            self.logger.warning("No strategy params found. cannot calculate grids.")
            return

        lower = self.strategy_params['lower_limit']
        upper = self.strategy_params['upper_limit']
        if self.grid_levels <= 0:
            self.logger.error("grid_levels must be > 0 to calculate grids.")
            return
        
        # In rolling mode, center grid around current price if outside bounds
        if self.rolling_grids_enabled and (current_price < lower or current_price > upper):
            range_size = upper - lower
            lower = current_price - (range_size / 2)
            upper = current_price + (range_size / 2)
            self.logger.info(f"Rolling mode: Centering grid [{lower:.2f}, {upper:.2f}] around {current_price}")
        elif current_price < lower or current_price > upper:
            self.logger.warning(f"Price {current_price} is out of bounds [{lower}, {upper}].")
            return

        self.grid_step = (upper - lower) / self.grid_levels
        
        self.grids = []
        for i in range(self.grid_levels + 1):
            price = lower + (i * self.grid_step)
            self.grids.append({
                'price': price,
                'orders': [],  # List of order dicts
            })
        self.logger.info(f"Calculated {len(self.grids)} grid levels, step={self.grid_step:.4f}")

    def place_order(self, grid_index, side, amount):
        """Helper to place an order and track it in the grid."""
        grid = self.grids[grid_index]
        price = grid['price']
        if self._has_existing_order(grid_index, side):
            self.logger.info(f"Skipping duplicate {side} order at grid {grid_index}")
            return False
        try:
            if side == 'buy':
                order = self.order_manager.create_limit_buy(self.symbol, amount, price)
            else:
                order = self.order_manager.create_limit_sell(self.symbol, amount, price)
            
            grid['orders'].append({'id': order['id'], 'side': side, 'amount': amount, 'price': price})
            self.active_orders.add(order['id'])
            self.logger.info(f"Placed {side} order at {price} (Grid {grid_index})")
            # Log PLACE event for ML training data
            self._log_grid_event(
                event_type='PLACE',
                side=side,
                price=price,
                amount=amount,
                grid_level=grid_index,
                order_id=order['id']
            )
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

    def check_orders(self, allow_rebalance=True):
        """Monitor open orders and handle fills."""
        try:
            if not self.strategy_params:
                return
            # Get all open orders to check against
            open_orders = self.order_manager.fetch_open_orders(self.symbol)
            open_ids = set(o['id'] for o in open_orders)

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
                            if not fetched:
                                self.logger.error(f"Order {order_id} not found on exchange.")
                                continue
                            status = (fetched.get('status') or '').lower()
                            filled = fetched.get('filled', 0.0)
                        except Exception as e:
                            self.logger.error(f"Failed to fetch order {order_id}: {e}")
                            continue

                        filled_amount = filled if filled and filled > 0 else 0.0
                        fill_price = fetched.get('average')
                        if fill_price is None:
                            fill_price = fetched.get('price')
                        if fill_price is None:
                            fill_price = order_data.get('price', grid.get('price'))

                        terminal_statuses = {'canceled', 'closed', 'expired', 'rejected'}
                        if status in terminal_statuses or filled_amount > 0:
                            grid['orders'].remove(order_data)
                            self.active_orders.discard(order_id)

                            if filled_amount > 0:
                                self.logger.info(f"Order {order_id} ({side}) filled qty={filled_amount}.")
                                self._apply_fill(side, filled_amount, fill_price)
                                # Log FILL event for ML training data
                                self._log_grid_event(
                                    event_type='FILL',
                                    side=side,
                                    price=fill_price,
                                    amount=filled_amount,
                                    grid_level=i,
                                    order_id=order_id,
                                    market_price=fill_price  # Use fill price as market price at fill time
                                )
                                self._rebalance_after_fill(i, side, filled_amount, allow_rebalance)
                            elif status == 'closed':
                                self.logger.warning(
                                    f"Order {order_id} closed with zero fill; no PnL/inventory update."
                                )
                            else:
                                self.logger.info(
                                    f"Order {order_id} reached terminal status {status}. Removing."
                                )
                            
        except Exception as e:
            self.logger.error(f"Error checking orders: {e}")

    def run(self):
        self.logger.info(f"Starting Grid Bot for {self.symbol}")
        pubsub = None
        
        while self.running:
            try:
                if pubsub is None:
                    pubsub = self.bus.subscribe(self.config['redis']['channels']['command'])

                # 1. Check for commands via Redis Streams (reliable) + Pub/Sub (legacy)
                self._check_stream_commands()
                msg = self.bus.get_message(pubsub)
                if msg:
                    self.handle_message(msg)

                # 2. Rate limit check before exchange API call
                if not self.rate_limiter.acquire(timeout=5.0):
                    self.logger.warning("Rate limit exhausted, skipping tick")
                    time.sleep(1)
                    continue

                # 3. Watchdog: Check for stale data (BEFORE fetching new)
                if self._check_stale_data():
                   self.logger.warning("Data is stale - attempting fresh fetch...")

                # 4. Fetch current price
                ticker = self.order_manager.fetch_ticker(self.symbol)
                last_price = ticker['last']
                self.last_price_update = time.time()
                
                # 5. Stop-Loss Check (CRITICAL SAFETY)
                if self._check_stop_loss(last_price):
                    continue  # Bot is stopping
                
                if not self.paused:
                    # Logic: If no grids, initialize
                    if not self.grids and self.strategy_params:
                        self.place_initial_orders(last_price)
                
                # Logic: Monitor orders (paused mode skips rebalancing)
                self.check_orders(allow_rebalance=not self.paused)

                # 6. Report Status
                self.report_status(last_price)
                
                time.sleep(2) # Throttle loop
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                pubsub = None
                time.sleep(5)

    def _check_stream_commands(self):
        """Check Redis Streams for reliable command delivery."""
        try:
            messages = self.bus.xreadgroup(
                COMMAND_GROUP, self.worker_id, COMMAND_STREAM, 
                count=10, block=1000  # Blocking 1s (prevents busy wait, allows heartbeats)
            )
            for msg_id, fields in messages:
                self.handle_message(fields)
                self.bus.xack(COMMAND_STREAM, COMMAND_GROUP, msg_id)
        except Exception as e:
            self.logger.error(f"Error reading stream commands: {e}")

    def _check_stale_data(self) -> bool:
        """
        Watchdog: Detect stale price data.
        Returns True if data is stale and we should skip this tick.
        """
        stale_seconds = time.time() - self.last_price_update
        if stale_seconds > STALE_DATA_THRESHOLD_SECONDS:
            self.logger.critical(
                f"STALE DATA DETECTED! Last update was {stale_seconds:.1f}s ago. "
                "Possible connection issue."
            )
            # Don't trade on stale data, but keep running to recover
            return True
        return False

    def _check_stop_loss(self, current_price) -> bool:
        """
        Check if price breached stop-loss threshold.
        Returns True if stop-loss triggered (bot is stopping).
        """
        if self.stop_loss_triggered:
            return True  # Already triggered, waiting to stop
            
        stop_loss = self.strategy_params.get('stop_loss')
        if stop_loss is None:
            return False
            
        if current_price <= stop_loss:
            self.logger.critical(
                f"🚨 STOP-LOSS TRIGGERED! Price {current_price} <= {stop_loss}"
            )
            self.logger.critical("Cancelling all orders and stopping bot...")
            
            self.stop_loss_triggered = True
            self.cancel_open_orders()
            self.running = False
            
            # Update status to reflect stop-loss
            self.db.update_worker_heartbeat(
                self.worker_id, self.symbol, 'STOP_LOSS', 0.0
            )
            self._publish_terminal_status('STOP_LOSS')  # Fix #2: Notify orchestrator
            return True
        return False

    def handle_message(self, msg):
        cmd = msg.get('command')
        target = msg.get('target')
        
        if target == 'all' or target == self.worker_id:
            if cmd == 'STOP':
                self.logger.warning("Received STOP signal.")
                self.cancel_open_orders()
                self.running = False
                self._publish_terminal_status('STOPPED')  # Fix #2: Notify orchestrator
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
        status = 'PAUSED' if self.paused else 'RUNNING'
        status_msg = {
            'worker_id': self.worker_id,
            'symbol': self.symbol,
            'status': status,
            'price': current_price,
            'inventory': self.inventory,
            'avg_cost': self.avg_cost,
            'realized_profit': self.realized_profit,
            'exposure': self.inventory * current_price,
            'last_updated': time.time()
        }
        if not self.bus.hset('workers:data', self.worker_id, json.dumps(status_msg)):
            self.logger.error("Failed to update worker status snapshot in Redis.")
        if not self.bus.publish(self.config['redis']['channels']['status'], status_msg):
            self.logger.error("Failed to publish worker status update.")

    def _publish_terminal_status(self, status: str):
        """Publish terminal status (STOPPED, STOP_LOSS, ERROR) to orchestrator for cleanup."""
        status_msg = {
            'worker_id': self.worker_id,
            'symbol': self.symbol,
            'status': status,
            'inventory': self.inventory,
            'avg_cost': self.avg_cost,
            'realized_profit': self.realized_profit,
            'last_updated': time.time()
        }
        if not self.bus.publish(self.config['redis']['channels']['status'], status_msg):
            self.logger.error(f"Failed to publish terminal status {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', required=True, help='Trading Pair (e.g. SOL/USDT)')
    parser.add_argument('--grids', type=int, default=20, help='Number of grid lines')
    args = parser.parse_args()

    bot = GridBot(args.pair, args.grids)
    bot.run()
