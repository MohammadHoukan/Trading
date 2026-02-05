
import unittest
import json
from unittest.mock import MagicMock
import sys
import os

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manager.risk_engine import RiskEngine, DrawdownAction
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

    def test_update_worker_params_succeeds_when_pubsub_fallback_works(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()

        orch.bus.xadd.return_value = None
        orch.bus.publish.return_value = True

        ok = orch.update_worker_params('w1', {'grid_levels': 8})

        self.assertTrue(ok)
        orch.bus.xadd.assert_called_once_with(
            'swarm:commands',
            {'command': 'UPDATE_PARAMS', 'target': 'w1', 'params': {'grid_levels': 8}},
        )
        orch.bus.publish.assert_called_once_with(
            'cmd',
            {'command': 'UPDATE_PARAMS', 'target': 'w1', 'params': {'grid_levels': 8}},
        )

    def test_update_worker_params_fails_when_stream_and_pubsub_fail(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()

        orch.bus.xadd.return_value = None
        orch.bus.publish.return_value = False

        ok = orch.update_worker_params('w1', {'grid_levels': 8})

        self.assertFalse(ok)

    def test_update_worker_scale_succeeds_when_pubsub_fallback_works(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()

        orch.bus.xadd.return_value = None
        orch.bus.publish.return_value = True

        ok = orch.update_worker_scale('w1', 0.5)

        self.assertTrue(ok)
        orch.bus.xadd.assert_called_once_with(
            'swarm:commands',
            {'command': 'UPDATE_SCALE', 'target': 'w1', 'scale': 0.5},
        )
        orch.bus.publish.assert_called_once_with(
            'cmd',
            {'command': 'UPDATE_SCALE', 'target': 'w1', 'scale': 0.5},
        )

    def test_update_worker_scale_rejects_non_positive_scale(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()

        self.assertFalse(orch.update_worker_scale('w1', 0.0))
        self.assertFalse(orch.update_worker_scale('w1', -1.0))
        orch.bus.xadd.assert_not_called()
        orch.bus.publish.assert_not_called()

    def test_drawdown_halt_retries_when_stop_broadcast_fails(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()
        orch.stop_broadcast_sent = False
        orch.drawdown_paused_workers = set()
        orch.last_drawdown_action = DrawdownAction.NORMAL
        orch.risk_engine = MagicMock()
        orch.risk_engine.max_global_capital = 500.0
        orch.risk_engine.get_status.return_value = {'total_allocated': 0.0}
        orch.risk_engine.check_drawdown.return_value = DrawdownAction.HALT_ALL
        orch.broadcast_command = MagicMock(return_value=False)

        orch.perform_risk_checks()
        self.assertFalse(orch.stop_broadcast_sent)
        orch.broadcast_command.assert_called_once_with('STOP')

        orch.perform_risk_checks()
        self.assertEqual(orch.broadcast_command.call_count, 2)

    def test_drawdown_recovery_resumes_only_workers_paused_by_drawdown(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = MagicMock()
        orch.logger = MagicMock()
        orch.stop_broadcast_sent = False
        orch.drawdown_paused_workers = set()
        orch.last_drawdown_action = DrawdownAction.NORMAL
        orch.risk_engine = MagicMock()
        orch.risk_engine.max_global_capital = 500.0
        orch.risk_engine.get_status.return_value = {'total_allocated': 0.0}
        orch.risk_engine.check_drawdown.side_effect = [
            DrawdownAction.REDUCE_EXPOSURE,
            DrawdownAction.NORMAL,
        ]
        orch.broadcast_command = MagicMock(return_value=True)
        orch.bus.hgetall.side_effect = [
            {
                'w1': json.dumps({'status': 'RUNNING'}),
                'w2': json.dumps({'status': 'PAUSED'}),
            },
            {
                'w1': json.dumps({'status': 'PAUSED'}),
                'w2': json.dumps({'status': 'PAUSED'}),
            },
        ]

        orch.perform_risk_checks()
        self.assertEqual(orch.drawdown_paused_workers, {'w1'})

        orch.perform_risk_checks()
        self.assertEqual(orch.drawdown_paused_workers, set())
        self.assertEqual(
            orch.broadcast_command.call_args_list,
            [
                unittest.mock.call('PAUSE', target='w1'),
                unittest.mock.call('RESUME', target='w1'),
            ],
        )


class TestDrawdownProtection(unittest.TestCase):
    """Tests for portfolio-level drawdown protection."""

    def setUp(self):
        self.config = {
            'swarm': {
                'risk_per_bot': 100.0,
                'max_global_capital': 500.0,
                'max_concurrency': 5
            },
            'risk': {
                'drawdown': {
                    'warning_pct': 10.0,
                    'reduce_pct': 15.0,
                    'halt_pct': 20.0,
                    'scale_factor': 0.5
                }
            }
        }
        self.risk_engine = RiskEngine(self.config)

    def test_drawdown_calculation_basic(self):
        """Test basic drawdown percentage calculation."""
        # Start with positive equity
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=0.0)
        self.assertEqual(self.risk_engine.peak_equity, 100.0)
        self.assertEqual(self.risk_engine.current_equity, 100.0)
        self.assertEqual(self.risk_engine.get_drawdown_pct(), 0.0)

        # Now lose some money
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-20.0)
        self.assertEqual(self.risk_engine.current_equity, 80.0)
        self.assertEqual(self.risk_engine.peak_equity, 100.0)  # Peak unchanged
        self.assertEqual(self.risk_engine.get_drawdown_pct(), 20.0)  # 20% drawdown

    def test_drawdown_warning_threshold(self):
        """Test that warning is triggered at 10% drawdown."""
        from manager.risk_engine import DrawdownAction

        # Build up equity
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=0.0)

        # 9% drawdown - should be normal
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-9.0)
        action = self.risk_engine.check_drawdown()
        self.assertEqual(action, DrawdownAction.NORMAL)

        # 11% drawdown - should warn but still normal action
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-11.0)
        action = self.risk_engine.check_drawdown()
        self.assertEqual(action, DrawdownAction.NORMAL)  # Warning doesn't change action

    def test_drawdown_reduce_threshold(self):
        """Test that REDUCE_EXPOSURE is triggered at 15% drawdown."""
        from manager.risk_engine import DrawdownAction

        # Build up equity
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=0.0)

        # 16% drawdown - should trigger reduce
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-16.0)
        action = self.risk_engine.check_drawdown()
        self.assertEqual(action, DrawdownAction.REDUCE_EXPOSURE)
        self.assertEqual(self.risk_engine.get_position_scale(), 0.5)

    def test_drawdown_halt_threshold(self):
        """Test that HALT_ALL is triggered at 20% drawdown."""
        from manager.risk_engine import DrawdownAction

        # Build up equity
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=0.0)

        # 21% drawdown - should trigger halt
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-21.0)
        action = self.risk_engine.check_drawdown()
        self.assertEqual(action, DrawdownAction.HALT_ALL)
        self.assertEqual(self.risk_engine.get_position_scale(), 0.0)

    def test_drawdown_recovery(self):
        """Test that action returns to NORMAL after recovery."""
        from manager.risk_engine import DrawdownAction

        # Build up equity and then lose
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=0.0)
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-16.0)
        action = self.risk_engine.check_drawdown()
        self.assertEqual(action, DrawdownAction.REDUCE_EXPOSURE)

        # Recover - unrealized goes back up
        self.risk_engine.update_worker_pnl('w1', realized_pnl=100.0, unrealized_pnl=-5.0)
        action = self.risk_engine.check_drawdown()
        self.assertEqual(action, DrawdownAction.NORMAL)

    def test_peak_equity_updates_on_new_high(self):
        """Test that peak equity updates when equity reaches new high."""
        self.risk_engine.update_worker_pnl('w1', realized_pnl=50.0, unrealized_pnl=0.0)
        self.assertEqual(self.risk_engine.peak_equity, 50.0)

        self.risk_engine.update_worker_pnl('w1', realized_pnl=75.0, unrealized_pnl=0.0)
        self.assertEqual(self.risk_engine.peak_equity, 75.0)

        # Losing doesn't change peak
        self.risk_engine.update_worker_pnl('w1', realized_pnl=75.0, unrealized_pnl=-10.0)
        self.assertEqual(self.risk_engine.peak_equity, 75.0)

    def test_multiple_workers_aggregate_pnl(self):
        """Test that PnL from multiple workers is aggregated correctly."""
        self.risk_engine.update_worker_pnl('w1', realized_pnl=50.0, unrealized_pnl=10.0)
        self.risk_engine.update_worker_pnl('w2', realized_pnl=30.0, unrealized_pnl=5.0)

        # Total: 50 + 10 + 30 + 5 = 95
        self.assertEqual(self.risk_engine.current_equity, 95.0)
        self.assertEqual(self.risk_engine.peak_equity, 95.0)

    def test_cleanup_worker_pnl_recalculates(self):
        """Test that cleanup_worker_pnl properly recalculates totals."""
        self.risk_engine.update_worker_pnl('w1', realized_pnl=50.0, unrealized_pnl=0.0)
        self.risk_engine.update_worker_pnl('w2', realized_pnl=30.0, unrealized_pnl=0.0)
        self.assertEqual(self.risk_engine.current_equity, 80.0)

        # Remove w1
        self.risk_engine.cleanup_worker_pnl('w1')
        self.assertEqual(self.risk_engine.current_equity, 30.0)

    def test_drawdown_uses_baseline_when_equity_never_positive(self):
        """Initial losses should still trigger drawdown protection."""
        self.risk_engine.update_worker_pnl('w1', realized_pnl=-120.0, unrealized_pnl=0.0)
        self.assertEqual(self.risk_engine.peak_equity, 0.0)
        self.assertAlmostEqual(self.risk_engine.get_drawdown_pct(), 24.0, places=6)
        self.assertEqual(self.risk_engine.check_drawdown(), DrawdownAction.HALT_ALL)


if __name__ == '__main__':
    unittest.main()
