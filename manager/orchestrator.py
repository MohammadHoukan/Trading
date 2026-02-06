import time
import sys
import os
import logging
import json

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pandas_ta as ta

from shared.messaging import RedisBus
from shared.config import load_config, get_redis_params
from manager.risk_engine import RiskEngine, DrawdownAction
from manager.regime_filter import RegimeFilter
from manager.mean_reversion_filter import MeanReversionFilter
from workers.order_manager import OrderManager

COMMAND_STREAM = 'swarm:commands'

class Orchestrator:
    def __init__(self, config_path='config/settings.yaml'):
        # Load Config (with env substitution)
        self.config = load_config(config_path)
            
        self.bus = RedisBus(**get_redis_params(self.config))
        self.risk_engine = RiskEngine(self.config)
        self.regime_filter = RegimeFilter(self.config)
        self.mean_reversion_filter = MeanReversionFilter(self.config)
        self.logger = logging.getLogger("Manager")
        logging.basicConfig(level=logging.INFO)

        # ATR cache for volatility-scaled grids
        self.atr_cache = {}  # {symbol: {'atr': float, 'timestamp': float}}
        
        self.running = True
        self.regime_by_symbol = {}  # {symbol: regime} for multi-symbol awareness
        self.last_regime_check = 0
        self.rejected_workers = set()
        self.last_stop_time = 0.0
        # Track workers paused by drawdown logic so recovery resumes only those workers.
        self.drawdown_paused_workers = set()
        self.last_drawdown_action = DrawdownAction.NORMAL
        
        # Market Data Publisher
        self.last_market_publish = 0.0
        self.market_publish_interval = 2.0  # Publish prices every 2 seconds
        
        # Initialize OrderManager for fetching tickers
        # Use first key from pool or legacy single key
        if 'pool' in self.config['exchange']:
            for creds in self.config['exchange']['pool']:
                k = creds.get('api_key', '')
                s = creds.get('secret', '')
                if k and not k.startswith('${'):
                    self.order_manager = OrderManager(
                        self.config['exchange']['name'],
                        k, s,
                        testnet=(self.config['exchange'].get('mode') == 'testnet')
                    )
                    break
            else:
                self.order_manager = None
                self.logger.warning("No valid API keys in pool for market data publishing")
        else:
            self.order_manager = OrderManager(
                self.config['exchange']['name'],
                self.config['exchange'].get('api_key', ''),
                self.config['exchange'].get('secret', ''),
                testnet=(self.config['exchange'].get('mode') == 'testnet')
            )

    def handle_worker_update(self, msg):
        """Process a single worker status update and persist it for the dashboard."""
        worker_id = msg.get('worker_id')
        symbol = msg.get('symbol')
        status = msg.get('status', 'RUNNING')
        
        if not worker_id or not symbol:
            return
            
        # Check for terminal status to release resources
        if status in ('STOPPED', 'STOP_LOSS', 'ERROR'):
            # Persist terminal status to dashboard before cleanup (Fix #5)
            msg['last_updated'] = time.time()
            self.bus.hset('workers:data', worker_id, json.dumps(msg))

            self.risk_engine.unregister_worker(worker_id)
            self.risk_engine.cleanup_worker_pnl(worker_id)
            self.rejected_workers.discard(worker_id)
            self.drawdown_paused_workers.discard(worker_id)
            return

        accepted = self.risk_engine.register_worker(worker_id, symbol)
        if not accepted:
            # Avoid flooding command channel with duplicate STOPs for the same rejected worker.
            if worker_id not in self.rejected_workers:
                self.logger.warning(
                    f"Worker {worker_id} rejected by risk engine (capacity limit). Sending STOP."
                )
                self.broadcast_command('STOP', target=worker_id)
                self.rejected_workers.add(worker_id)
            return

        self.rejected_workers.discard(worker_id)

        if 'exposure' in msg:
            self.risk_engine.update_exposure(worker_id, msg['exposure'])

        # Update PnL tracking for drawdown calculation
        realized_pnl = msg.get('realized_profit', 0.0)
        inventory = msg.get('inventory', 0.0)
        avg_cost = msg.get('avg_cost', 0.0)
        price = msg.get('price', 0.0)
        unrealized_pnl = (price - avg_cost) * inventory if inventory > 0 and avg_cost > 0 else 0.0
        self.risk_engine.update_worker_pnl(worker_id, realized_pnl, unrealized_pnl)

        msg['last_updated'] = time.time()
        if not self.bus.hset('workers:data', worker_id, json.dumps(msg)):
            self.logger.error(f"Failed to persist worker snapshot for {worker_id}")

    def run(self):
        self.logger.info("Starting Orchestrator...")
        pubsub = None
        
        while self.running:
            try:
                # (Re)subscribe on startup and after Redis failures.
                if pubsub is None:
                    pubsub = self.bus.subscribe(self.config['redis']['channels']['status'])

                # 1. Listen for heartbeats
                msg = self.bus.get_message(pubsub)
                if msg:
                     self.logger.info(f"Worker update: {msg}")
                     self.handle_worker_update(msg)

                # 2. Risk Checks (Stub)
                self.perform_risk_checks()
                
                # 3. Regime Detection (Every 60s)
                now = time.time()
                if now - self.last_regime_check > 60:
                     self.perform_regime_checks()
                     self.last_regime_check = now

                # 4. Market Data Publishing (Every 2s)
                if now - self.last_market_publish > self.market_publish_interval:
                    self.publish_market_data()
                    self.last_market_publish = now

                time.sleep(1)

            except KeyboardInterrupt:
                self.shutdown()
            except Exception as e:
                self.logger.error(f"Orchestrator error: {e}")
                pubsub = None
                time.sleep(1)

    def perform_risk_checks(self):
        # Backward compatibility for tests that instantiate Orchestrator via __new__.
        if not hasattr(self, 'last_stop_time'):
            self.last_stop_time = 0.0
        if not hasattr(self, 'drawdown_paused_workers'):
            self.drawdown_paused_workers = set()
        if not hasattr(self, 'last_drawdown_action'):
            self.last_drawdown_action = DrawdownAction.NORMAL

        # Check global limits
        status = self.risk_engine.get_status()
        self.logger.debug(f"Risk Status: {status}")

        # 1. Check capital allocation limits
        is_global_risk = status['total_allocated'] > self.risk_engine.max_global_capital
        
        if is_global_risk:
            self.logger.critical("GLOBAL RISK LIMIT EXCEEDED!")
            self._ensure_stopped()

        # 2. Check drawdown limits
        drawdown_action = self.risk_engine.check_drawdown()

        if drawdown_action == DrawdownAction.HALT_ALL:
            # Critical drawdown - stop all trading
            self.logger.critical("DRAWDOWN HALT: Stopping all workers due to excessive drawdown!")
            self._ensure_stopped()

        elif drawdown_action == DrawdownAction.REDUCE_EXPOSURE:
            # Elevated drawdown - pause new orders but don't close positions
            # Workers in PAUSED state won't place new orders but will manage existing ones
            self._pause_all_workers_for_drawdown()

        elif (
            drawdown_action == DrawdownAction.NORMAL
            and self.last_drawdown_action == DrawdownAction.REDUCE_EXPOSURE
        ):
            # Recovery from drawdown reduction: resume only workers paused by drawdown logic.
            self._resume_workers_paused_for_drawdown()

        self.last_drawdown_action = drawdown_action

    def _get_active_symbols(self):
        """Get list of unique symbols from active workers."""
        try:
            worker_data = self.bus.hgetall('workers:data')
            symbols = set()
            for data in worker_data.values():
                import json
                try:
                    w = json.loads(data)
                    if w.get('status') in ('RUNNING', 'PAUSED'):
                        symbols.add(w.get('symbol'))
                except:
                    pass
            return list(symbols) if symbols else [self.regime_filter.default_symbol]
        except:
            return [self.regime_filter.default_symbol]

    def perform_regime_checks(self):
        """Check market regime for each active symbol and issue targeted scale updates."""
        active_symbols = self._get_active_symbols()
        
        for symbol in active_symbols:
            analysis = self.regime_filter.analyze_market(symbol)
            
            regime = analysis.get('regime', 'UNKNOWN')
            score = analysis.get('score', 50)
            recommendation = analysis.get('recommendation', 'HOLD')
            scale = analysis.get('scale', 1.0)
            
            self.logger.info(
                f"Regime Check [{symbol}]: {regime} (score={score}, rec={recommendation}, scale={scale})"
            )

            if regime in {'ERROR', 'UNKNOWN'}:
                self.logger.warning(f"Skipping regime transition for {symbol} due to {regime}")
                continue
            
            last_regime = self.regime_by_symbol.get(symbol)
            
            # Only act on state changes to avoid spamming commands
            if regime != last_regime:
                if recommendation == 'REDUCE_EXPOSURE':
                    self.logger.warning(
                        f"Regime change to {regime} for {symbol}. Reducing exposure to scale={scale}."
                    )
                    self.update_symbol_scale(symbol, scale)
                elif recommendation == 'RUN' and last_regime in {'TRENDING', 'UNCERTAIN'}:
                    self.logger.info(f"Regime change to {regime} for {symbol}. Restoring exposure scale=1.0.")
                    self.update_symbol_scale(symbol, 1.0)
                # HOLD = keep current state, don't send command
                
                self.regime_by_symbol[symbol] = regime

    def _pause_symbol_workers(self, symbol):
        """Send PAUSE to all workers trading a specific symbol."""
        try:
            worker_data = self.bus.hgetall('workers:data')
            for worker_id, data in worker_data.items():
                import json
                try:
                    w = json.loads(data)
                    if w.get('symbol') == symbol and w.get('status') == 'RUNNING':
                        self.broadcast_command('PAUSE', target=worker_id)
                except:
                    pass
        except Exception as e:
            self.logger.error(f"Failed to pause workers for {symbol}: {e}")

    def _resume_symbol_workers(self, symbol):
        """Send RESUME to all workers trading a specific symbol."""
        try:
            worker_data = self.bus.hgetall('workers:data')
            for worker_id, data in worker_data.items():
                import json
                try:
                    w = json.loads(data)
                    if w.get('symbol') == symbol and w.get('status') == 'PAUSED':
                        self.broadcast_command('RESUME', target=worker_id)
                except:
                    pass
        except Exception as e:
            self.logger.error(f"Failed to resume workers for {symbol}: {e}")

    def _pause_all_workers_for_drawdown(self):
        """Pause all running workers due to drawdown threshold breach."""
        try:
            worker_data = self.bus.hgetall('workers:data')
            if not worker_data:
                return
            paused_count = 0
            for worker_id, data in worker_data.items():
                try:
                    w = json.loads(data)
                    if w.get('status') == 'RUNNING' and worker_id not in self.drawdown_paused_workers:
                        if self.broadcast_command('PAUSE', target=worker_id):
                            self.drawdown_paused_workers.add(worker_id)
                            paused_count += 1
                except Exception:
                    pass
            if paused_count > 0:
                self.logger.warning(f"Paused {paused_count} workers due to elevated drawdown")
        except Exception as e:
            self.logger.error(f"Failed to pause workers for drawdown: {e}")

    def _resume_workers_paused_for_drawdown(self):
        """Resume only workers previously paused by drawdown protection."""
        if not self.drawdown_paused_workers:
            return
        try:
            worker_data = self.bus.hgetall('workers:data') or {}
            resumed_count = 0
            for worker_id in list(self.drawdown_paused_workers):
                raw = worker_data.get(worker_id)
                if not raw:
                    self.drawdown_paused_workers.discard(worker_id)
                    continue
                try:
                    status = json.loads(raw).get('status')
                except Exception:
                    continue
                if status in ('STOPPED', 'STOP_LOSS', 'ERROR'):
                    self.drawdown_paused_workers.discard(worker_id)
                elif status == 'PAUSED':
                    if self.broadcast_command('RESUME', target=worker_id):
                        self.drawdown_paused_workers.discard(worker_id)
                        resumed_count += 1
                else:
                    # Already running or in another transient state; stop tracking.
                    self.drawdown_paused_workers.discard(worker_id)
            if resumed_count > 0:
                self.logger.info(f"Resumed {resumed_count} workers after drawdown recovery")
        except Exception as e:
            self.logger.error(f"Failed to resume drawdown-paused workers: {e}")

    def _ensure_stopped(self):
        """
        Continuously assert the STOP command while a critical condition persists.
        Throttled to once per second to avoid flooding.
        """
        now = time.time()
        if now - self.last_stop_time >= 1.0:
            self.logger.warning("Asserting GLOBAL STOP due to critical risk state.")
            if self.broadcast_command('STOP'):
                self.last_stop_time = now
            else:
                self.logger.error("Failed to broadcast STOP assertion! (Will retry)")

    def broadcast_command(self, cmd, target='all'):
        """Broadcast command to workers. Returns True on success, False on failure."""
        message = {'command': cmd, 'target': target}
        stream_id = self.bus.xadd(COMMAND_STREAM, message)
        stream_success = stream_id is not None
        if not stream_success:
            self.logger.error(f"Failed to enqueue command {cmd} to stream for {target}")

        pubsub_success = self.bus.publish(self.config['redis']['channels']['command'], message)
        if not pubsub_success:
            self.logger.error(f"Failed to broadcast command {cmd} to {target}")

        return stream_success or pubsub_success

    def update_worker_params(self, worker_id: str, params: dict):
        """
        Send parameter update to a specific worker.

        Args:
            worker_id: Target worker ID
            params: Dict of parameters to update (e.g., lower_limit, upper_limit, grid_levels)

        Returns:
            True on success, False on failure
        """
        if not params:
            self.logger.warning("update_worker_params called with empty params")
            return False

        message = {
            'command': 'UPDATE_PARAMS',
            'target': worker_id,
            'params': params
        }

        stream_id = self.bus.xadd(COMMAND_STREAM, message)
        stream_success = stream_id is not None
        if not stream_success:
            self.logger.error(f"Failed to enqueue UPDATE_PARAMS to stream for {worker_id}")

        pubsub_success = self.bus.publish(self.config['redis']['channels']['command'], message)
        if not pubsub_success:
            self.logger.error(f"Failed to publish UPDATE_PARAMS for {worker_id}")

        if stream_success or pubsub_success:
            self.logger.info(f"Sent UPDATE_PARAMS to {worker_id}: {params}")
            return True
        return False

    def update_worker_scale(self, worker_id: str, scale: float):
        """
        Send exposure-scale update to a specific worker.

        Args:
            worker_id: Target worker ID
            scale: Multiplicative scale for amount_per_grid (e.g., 0.5 = half size)

        Returns:
            True on success, False on failure
        """
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            self.logger.error(f"Invalid scale for {worker_id}: {scale}")
            return False

        if scale <= 0:
            self.logger.error(f"Scale must be > 0 for {worker_id}: {scale}")
            return False

        message = {
            'command': 'UPDATE_SCALE',
            'target': worker_id,
            'scale': scale,
        }

        stream_id = self.bus.xadd(COMMAND_STREAM, message)
        stream_success = stream_id is not None
        if not stream_success:
            self.logger.error(f"Failed to enqueue UPDATE_SCALE to stream for {worker_id}")

        pubsub_success = self.bus.publish(self.config['redis']['channels']['command'], message)
        if not pubsub_success:
            self.logger.error(f"Failed to publish UPDATE_SCALE for {worker_id}")

        if stream_success or pubsub_success:
            self.logger.info(f"Sent UPDATE_SCALE to {worker_id}: scale={scale}")
            return True
        return False

    def update_symbol_scale(self, symbol: str, scale: float):
        """
        Send exposure-scale update to all workers trading a specific symbol.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            scale: Multiplicative scale factor for amount_per_grid

        Returns:
            Number of workers updated
        """
        updated = 0
        try:
            worker_data = self.bus.hgetall('workers:data')
            for worker_id, data in worker_data.items():
                try:
                    w = json.loads(data)
                    if w.get('symbol') == symbol and w.get('status') in ('RUNNING', 'PAUSED'):
                        if self.update_worker_scale(worker_id, scale):
                            updated += 1
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"Failed to update scale for {symbol}: {e}")

        self.logger.info(f"Updated scale={scale} for {updated} workers on {symbol}")
        return updated

    def update_symbol_params(self, symbol: str, params: dict):
        """
        Send parameter update to all workers trading a specific symbol.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            params: Dict of parameters to update

        Returns:
            Number of workers updated
        """
        updated = 0
        try:
            worker_data = self.bus.hgetall('workers:data')
            for worker_id, data in worker_data.items():
                try:
                    w = json.loads(data)
                    if w.get('symbol') == symbol and w.get('status') in ('RUNNING', 'PAUSED'):
                        if self.update_worker_params(worker_id, params):
                            updated += 1
                except Exception:
                    pass
        except Exception as e:
            self.logger.error(f"Failed to update params for {symbol}: {e}")

        self.logger.info(f"Updated {updated} workers for {symbol}")
        return updated

    def publish_market_data(self):
        """
        Fetch current prices for all active symbols and broadcast via Redis.
        Workers subscribe to 'market_data:{symbol}' channels.

        Also calculates and broadcasts ATR for volatility-scaled grids.
        """
        if not self.order_manager:
            return

        active_symbols = self._get_active_symbols()
        if not active_symbols:
            return

        for symbol in active_symbols:
            try:
                ticker = self.order_manager.fetch_ticker(symbol)
                if ticker and ticker.get('last'):
                    current_price = ticker['last']

                    # Calculate ATR for volatility scaling
                    atr = self._calculate_atr(symbol)
                    atr_pct = (atr / current_price) if atr and current_price else None

                    payload = {
                        'symbol': symbol,
                        'price': current_price,
                        'bid': ticker.get('bid'),
                        'ask': ticker.get('ask'),
                        'atr': atr,
                        'atr_pct': atr_pct,
                        'timestamp': time.time()
                    }
                    channel = f"market_data:{symbol.replace('/', '_')}"
                    self.bus.publish(channel, payload)
                    self.logger.debug(
                        f"Published market data for {symbol}: price={current_price}, atr={atr}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to fetch/publish market data for {symbol}: {e}")

    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Average True Range for a symbol.

        Uses caching to avoid excessive API calls.

        Args:
            symbol: Trading pair
            period: ATR period (default 14)

        Returns:
            ATR value or None if unavailable
        """
        now = time.time()

        # Check cache (ATR doesn't change rapidly, cache for 60 seconds)
        if symbol in self.atr_cache:
            cached = self.atr_cache[symbol]
            if now - cached['timestamp'] < 60:
                return cached['atr']

        try:
            # Fetch OHLCV data
            candles = self.order_manager.exchange.fetch_ohlcv(
                symbol, '1h', limit=period + 10
            )
            if not candles or len(candles) < period:
                return None

            df = pd.DataFrame(
                candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Calculate ATR using pandas_ta
            atr = df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=period)

            if atr is None or atr.empty:
                return None

            atr_value = atr.iloc[-1]
            if pd.isna(atr_value):
                return None

            # Cache the result
            self.atr_cache[symbol] = {
                'atr': float(atr_value),
                'timestamp': now
            }

            return float(atr_value)

        except Exception as e:
            self.logger.warning(f"Failed to calculate ATR for {symbol}: {e}")
            return None

    def shutdown(self):
        self.logger.info("Shutting down... sending STOP to all workers.")
        self.broadcast_command('STOP')
        self.running = False

if __name__ == "__main__":
    manager = Orchestrator()
    manager.run()
