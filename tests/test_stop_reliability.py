
import unittest
from unittest.mock import MagicMock, patch
import time
import sys
import os

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manager.orchestrator import Orchestrator
from manager.risk_engine import DrawdownAction

class TestStopReliability(unittest.TestCase):
    def setUp(self):
        # Setup minimal orchestrator with mocks
        self.orch = Orchestrator.__new__(Orchestrator)
        self.orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        self.orch.bus = MagicMock()
        self.orch.risk_engine = MagicMock()
        self.orch.logger = MagicMock()
        
        # Initialize new state
        self.orch.last_stop_time = 0.0
        self.orch.risk_engine.max_global_capital = 100.0

    def test_ensure_stopped_retries_on_throttle(self):
        """Verify _ensure_stopped broadcasts STOP only after throttle delay."""
        # 1. Initial Call (Should Broadcast)
        with patch('time.time', return_value=1000.0):
            self.orch._ensure_stopped()
            self.orch.bus.publish.assert_called_with('cmd', {'command': 'STOP', 'target': 'all'})
            self.assertEqual(self.orch.last_stop_time, 1000.0)
        
        # Reset Mock
        self.orch.bus.publish.reset_mock()
        
        # 2. Call within 0.5s (Should NOT Broadcast)
        with patch('time.time', return_value=1000.5):
            self.orch._ensure_stopped()
            self.orch.bus.publish.assert_not_called()
            
        # 3. Call after 1.1s (Should Broadcast)
        with patch('time.time', return_value=1001.1):
            self.orch._ensure_stopped()
            self.orch.bus.publish.assert_called_with('cmd', {'command': 'STOP', 'target': 'all'})
            self.assertEqual(self.orch.last_stop_time, 1001.1)

    def test_global_risk_limit_continuously_asserts_stop(self):
        """Verify GLOBAL RISK branch calls _ensure_stopped."""
        # Setup: Risk Breached
        self.orch.risk_engine.get_status.return_value = {'total_allocated': 200.0}
        self.orch.risk_engine.check_drawdown.return_value = DrawdownAction.NORMAL
        
        # Calls _ensure_stopped internally
        with patch.object(self.orch, '_ensure_stopped') as mock_ensure:
            self.orch.perform_risk_checks()
            mock_ensure.assert_called_once()

    def test_drawdown_halt_continuously_asserts_stop(self):
        """Verify DRAWDOWN HALT branch calls _ensure_stopped."""
        # Setup: Allocation OK, but Drawdown HALT
        self.orch.risk_engine.get_status.return_value = {'total_allocated': 50.0}
        self.orch.risk_engine.check_drawdown.return_value = DrawdownAction.HALT_ALL
        
        # Calls _ensure_stopped internally
        with patch.object(self.orch, '_ensure_stopped') as mock_ensure:
            self.orch.perform_risk_checks()
            mock_ensure.assert_called_once()

if __name__ == '__main__':
    unittest.main()
