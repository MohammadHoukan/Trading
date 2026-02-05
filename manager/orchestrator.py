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
        self.last_regime = None
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

    def perform_regime_checks(self):
        regime = self.regime_filter.analyze_market()
        self.logger.info(f"Regime Check: {regime}")

        if regime in {'ERROR', 'UNKNOWN'}:
            self.logger.warning(f"Skipping regime transition due to {regime}")
            return
        
        if regime != self.last_regime:
            if regime == 'TRENDING':
                self.logger.warning("Regime change to TRENDING. Pausing Swarm.")
                self.broadcast_command('PAUSE')
            elif regime == 'RANGING' and self.last_regime == 'TRENDING':
                self.logger.info("Regime change to RANGING. Resuming Swarm.")
                self.broadcast_command('RESUME')
            
            self.last_regime = regime

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
