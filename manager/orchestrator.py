import time
import sys
import os
import logging
import json

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.messaging import RedisBus
from shared.config import load_config, get_redis_params
from shared.config import load_config, get_redis_params
from manager.risk_engine import RiskEngine
from manager.regime_filter import RegimeFilter

class Orchestrator:
    def __init__(self, config_path='config/settings.yaml'):
        # Load Config (with env substitution)
        self.config = load_config(config_path)
            
        self.bus = RedisBus(**get_redis_params(self.config))
        self.bus = RedisBus(**get_redis_params(self.config))
        self.risk_engine = RiskEngine(self.config)
        self.regime_filter = RegimeFilter(self.config)
        self.logger = logging.getLogger("Manager")
        logging.basicConfig(level=logging.INFO)
        
        self.running = True
        self.last_regime = None
        self.last_regime_check = 0

    def run(self):
        self.logger.info("Starting Orchestrator...")
        
        # Subscribe to worker status updates
        pubsub = self.bus.subscribe(self.config['redis']['channels']['status'])
        
        while self.running:
            try:
                # 1. Listen for heartbeats
                msg = self.bus.get_message(pubsub)
                if msg:
                     self.logger.info(f"Worker update: {msg}")
                     # Update Risk Engine
                     if 'worker_id' in msg and 'symbol' in msg:
                         self.risk_engine.register_worker(msg['worker_id'], msg['symbol'])
                         
                         if 'exposure' in msg:
                             self.risk_engine.update_exposure(msg['worker_id'], msg['exposure'])

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

    def perform_risk_checks(self):
        # Check global limits
        status = self.risk_engine.get_status()
        self.logger.debug(f"Risk Status: {status}")
        
        if status['total_allocated'] > self.risk_engine.max_global_capital:
            self.logger.critical("GLOBAL RISK LIMIT EXCEEDED! STOPPING ALL WORKERS.")
            self.broadcast_command('STOP')

    def perform_regime_checks(self):
        regime = self.regime_filter.analyze_market()
        self.logger.info(f"Regime Check: {regime}")
        
        if regime != self.last_regime:
            if regime == 'TRENDING':
                self.logger.warning("Regime change to TRENDING. Pausing Swarm.")
                self.broadcast_command('PAUSE')
            elif regime == 'RANGING' and self.last_regime == 'TRENDING':
                self.logger.info("Regime change to RANGING. Resuming Swarm.")
                self.broadcast_command('RESUME')
            
            self.last_regime = regime

    def broadcast_command(self, cmd, target='all'):
        message = {'command': cmd, 'target': target}
        self.bus.publish(self.config['redis']['channels']['command'], message)

    def shutdown(self):
        self.logger.info("Shutting down... sending STOP to all workers.")
        self.broadcast_command('STOP')
        self.running = False

if __name__ == "__main__":
    manager = Orchestrator()
    manager.run()
