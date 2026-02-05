
import unittest
from unittest.mock import MagicMock
import sys
import os

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manager.risk_engine import RiskEngine
from manager.orchestrator import Orchestrator

class TestRiskEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.config = {
            'swarm': {
                'risk_per_bot': 100.0,
                'max_global_capital': 500.0,
                'max_concurrency': 5
            }
        }
        self.risk_engine = RiskEngine(self.config)

    def test_risk_limits(self):
        # 1. Register Worker
        self.assertTrue(self.risk_engine.register_worker('w1', 'SOL/USDT'))
        
        # 2. Update Exposure below limit
        self.risk_engine.update_exposure('w1', 50.0)
        status = self.risk_engine.get_status()
        self.assertEqual(status['total_allocated'], 50.0)
        
        # 3. Exceed Global Limit
        # Set w1 to 600 (Limit 500)
        # Note: update_exposure is passive updates from bot, so it accepts it 
        # but orchestartor checks total > limit.
        self.risk_engine.update_exposure('w1', 600.0)
        status = self.risk_engine.get_status()
        self.assertEqual(status['total_allocated'], 600.0)
        self.assertTrue(status['total_allocated'] > self.risk_engine.max_global_capital)

    def test_orchestrator_reaction(self):
        # Mock Orchestrator
        orch = Orchestrator()
        orch.config = {'redis': {'channels': {'command': 'cmd'}}} # Minimally sufficient
        orch.bus = MagicMock()
        orch.risk_engine = self.risk_engine
        
        # Limits: Global 500.
        
        # Scenario: Worker reports 600 exposure.
        orch.risk_engine.update_exposure('w1', 600.0)
        
        # Run Risk Checks
        orch.perform_risk_checks()
        
        # Expectation: Broadcast STOP
        orch.bus.publish.assert_called_with('cmd', {'command': 'STOP', 'target': 'all'})

if __name__ == '__main__':
    unittest.main()
