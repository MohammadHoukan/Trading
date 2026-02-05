import time
import sys
import os
import logging
import json

# Add root directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.messaging import RedisBus
from shared.config import load_config

class Orchestrator:
    def __init__(self, config_path='config/settings.yaml'):
        # Load Config (with env substitution)
        self.config = load_config(config_path)
            
        redis_cfg = self.config['redis']
        self.bus = RedisBus(host=redis_cfg['host'], port=redis_cfg['port'], db=redis_cfg['db'])
        self.logger = logging.getLogger("Manager")
        logging.basicConfig(level=logging.INFO)
        
        self.running = True

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

                # 2. Risk Checks (Stub)
                self.perform_risk_checks()
                
                # 3. Regime Detection (Stub)
                # if market_crash: broadcast_kill_switch()

                time.sleep(1)

            except KeyboardInterrupt:
                self.shutdown()
            except Exception as e:
                self.logger.error(f"Orchestrator error: {e}")

    def perform_risk_checks(self):
        # Example: Check total exposure across all bots
        pass

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
