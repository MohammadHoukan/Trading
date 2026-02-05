
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
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}} # Minimally sufficient
        orch.bus = MagicMock()
        orch.risk_engine = self.risk_engine
        orch.logger = MagicMock()
        orch.stop_broadcast_sent = False
        
        # Limits: Global 500.
        
        # Scenario: Worker reports 600 exposure.
        orch.risk_engine.update_exposure('w1', 600.0)
        
        # Run Risk Checks
        orch.perform_risk_checks()
        
        # Expectation: Broadcast STOP
        orch.bus.publish.assert_called_with('cmd', {'command': 'STOP', 'target': 'all'})

    def test_orchestrator_rejects_excess_worker_updates(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.risk_engine = MagicMock()
        orch.logger = MagicMock()
        orch.rejected_workers = set()

        orch.risk_engine.register_worker.return_value = False

        msg = {
            'worker_id': 'w_rejected',
            'symbol': 'ETH/USDT',
            'exposure': 75.0,
        }

        orch.handle_worker_update(msg)

        orch.risk_engine.update_exposure.assert_not_called()
        orch.bus.hset.assert_not_called()
        orch.bus.publish.assert_called_once_with('cmd', {'command': 'STOP', 'target': 'w_rejected'})

        # Duplicate updates for the same worker should not spam STOP repeatedly.
        orch.handle_worker_update(msg)
        orch.bus.publish.assert_called_once()

    def test_orchestrator_logs_on_worker_snapshot_failure(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.risk_engine = MagicMock()
        orch.logger = MagicMock()
        orch.rejected_workers = set()

        orch.risk_engine.register_worker.return_value = True
        orch.bus.hset.return_value = False

        msg = {
            'worker_id': 'w1',
            'symbol': 'SOL/USDT',
            'exposure': 12.5,
        }
        orch.handle_worker_update(msg)

        orch.risk_engine.update_exposure.assert_called_once_with('w1', 12.5)
        orch.bus.hset.assert_called_once()
        orch.logger.error.assert_called_once_with("Failed to persist worker snapshot for w1")

    def test_orchestrator_retries_stop_on_sustained_breach(self):
        """Test that STOP is broadcast repeatedly while risk limit is exceeded."""
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.risk_engine = self.risk_engine
        orch.logger = MagicMock()
        orch.stop_broadcast_sent = False
        
        # 1. Breach Limits
        orch.risk_engine.update_exposure('w1', 600.0) # Limit is 500
        
        # 2. First Check -> Should Broadcast
        orch.bus.publish.return_value = True
        orch.perform_risk_checks()
        orch.bus.publish.assert_called_with('cmd', {'command': 'STOP', 'target': 'all'})
        self.assertTrue(orch.stop_broadcast_sent)
        
        # Reset mock to check next call
        orch.bus.publish.reset_mock()
        
        # 3. Second Check (Still Breached) -> Should Broadcast AGAIN (Fix #1)
        orch.perform_risk_checks()
        orch.bus.publish.assert_called_with('cmd', {'command': 'STOP', 'target': 'all'})

    def test_broadcast_command_succeeds_when_stream_write_succeeds(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()

        orch.bus.xadd.return_value = "1-0"
        orch.bus.publish.return_value = False

        ok = orch.broadcast_command('STOP', target='all')

        self.assertTrue(ok)
        orch.bus.xadd.assert_called_once_with('swarm:commands', {'command': 'STOP', 'target': 'all'})
        orch.bus.publish.assert_called_once_with('cmd', {'command': 'STOP', 'target': 'all'})

    def test_broadcast_command_fails_when_stream_and_pubsub_fail(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()

        orch.bus.xadd.return_value = None
        orch.bus.publish.return_value = False

        ok = orch.broadcast_command('STOP', target='all')

        self.assertFalse(ok)

if __name__ == '__main__':
    unittest.main()
