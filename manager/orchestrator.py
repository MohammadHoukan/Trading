import time
import sys
import os
import logging
import json

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.messaging import RedisBus
from shared.config import load_config, get_redis_params
from manager.risk_engine import RiskEngine
from manager.regime_filter import RegimeFilter

class Orchestrator:
    def __init__(self, config_path='config/settings.yaml'):
        # Load Config (with env substitution)
        self.config = load_config(config_path)
            
        self.bus = RedisBus(**get_redis_params(self.config))
        self.risk_engine = RiskEngine(self.config)
        self.regime_filter = RegimeFilter(self.config)
        self.logger = logging.getLogger("Manager")
        logging.basicConfig(level=logging.INFO)
        
        self.running = True
        self.regime_by_symbol = {}  # {symbol: regime} for multi-symbol awareness
        self.last_regime_check = 0
        self.rejected_workers = set()
        self.stop_broadcast_sent = False

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
            self.rejected_workers.discard(worker_id)
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

                time.sleep(1)

            except KeyboardInterrupt:
                self.shutdown()
            except Exception as e:
                self.logger.error(f"Orchestrator error: {e}")
                pubsub = None
                time.sleep(1)

    def perform_risk_checks(self):
        # Check global limits
        status = self.risk_engine.get_status()
        self.logger.debug(f"Risk Status: {status}")
        
        if status['total_allocated'] > self.risk_engine.max_global_capital:
            # Fix #1: Retry STOP command while breach persists (don't silence after one attempt)
            self.logger.critical("GLOBAL RISK LIMIT EXCEEDED! STOPPING ALL WORKERS.")
            if self.broadcast_command('STOP'):
                self.stop_broadcast_sent = True
            else:
                self.logger.error("Failed to broadcast STOP command! Will retry next tick.")
        else:
            if self.stop_broadcast_sent:
                 self.logger.info("Global risk normalization. Resetting STOP flag.")
                 self.stop_broadcast_sent = False

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
        """Check market regime for each active symbol and issue targeted PAUSE/RESUME."""
        active_symbols = self._get_active_symbols()
        
        for symbol in active_symbols:
            analysis = self.regime_filter.analyze_market(symbol)
            
            regime = analysis.get('regime', 'UNKNOWN')
            score = analysis.get('score', 50)
            recommendation = analysis.get('recommendation', 'HOLD')
            
            self.logger.info(f"Regime Check [{symbol}]: {regime} (score={score}, rec={recommendation})")

            if regime in {'ERROR', 'UNKNOWN'}:
                self.logger.warning(f"Skipping regime transition for {symbol} due to {regime}")
                continue
            
            last_regime = self.regime_by_symbol.get(symbol)
            
            # Only act on state changes to avoid spamming commands
            if regime != last_regime:
                if recommendation == 'PAUSE':
                    self.logger.warning(f"Regime change to {regime} for {symbol}. Pausing.")
                    self._pause_symbol_workers(symbol)
                elif recommendation == 'RUN' and last_regime in {'TRENDING', 'UNCERTAIN'}:
                    self.logger.info(f"Regime change to {regime} for {symbol}. Resuming.")
                    self._resume_symbol_workers(symbol)
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

    def broadcast_command(self, cmd, target='all'):
        """Broadcast command to workers. Returns True on success, False on failure."""
        message = {'command': cmd, 'target': target}
        success = self.bus.publish(self.config['redis']['channels']['command'], message)
        if not success:
            self.logger.error(f"Failed to broadcast command {cmd} to {target}")
        return success

    def shutdown(self):
        self.logger.info("Shutting down... sending STOP to all workers.")
        self.broadcast_command('STOP')
        self.running = False

if __name__ == "__main__":
    manager = Orchestrator()
    manager.run()
