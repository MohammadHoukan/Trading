import py_compile
import unittest
from unittest.mock import MagicMock, patch
import json
import threading
import time
import sys
import os

# Add root path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manager.orchestrator import Orchestrator

class TestDashboardFlow(unittest.TestCase):
    @patch('manager.orchestrator.RedisBus')
    @patch('manager.orchestrator.load_config')
    @patch('manager.orchestrator.get_redis_params')
    @patch('manager.risk_engine.RiskEngine')
    @patch('manager.regime_filter.RegimeFilter')
    def test_worker_persistence(self, MockRegime, MockRisk, MockGetParams, MockLoadConfig, MockBus):
        # Setup Mocks
        mock_config = {
            'redis': {
                'channels': {'status': 'swarm:status', 'command': 'swarm:cmd'},
                'host': 'localhost', 'port': 6379, 'db': 0
            },
            'risk': {'max_global_capital': 1000},
            'swarm': {'risk_per_bot': 100.0},
            'exchange': {'name': 'binance', 'mode': 'testnet', 'api_key': 'test', 'secret': 'test'}
        }
        MockLoadConfig.return_value = mock_config
        MockGetParams.return_value = {}
        
        # Mock Redis Bus and Client
        mock_bus_instance = MockBus.return_value
        mock_bus_instance.hset.return_value = True

        # Mock PubSub
        mock_pubsub = MagicMock()
        mock_bus_instance.subscribe.return_value = mock_pubsub
        
        # Orchestrator Instance
        orch = Orchestrator()
        
        # Simulate RUNNING loop for one iteration
        orch.running = True
        
        # Mock getting a message
        worker_msg = {
            'worker_id': 'bench_worker', 
            'symbol': 'ETH/USDT',
            'exposure': 50,
            'pnl': 10
        }
        
        # First call returns message, second call raises StopIteration to break loop (or we just manually break)
        mock_bus_instance.get_message.side_effect = [worker_msg, None]
        
        # We need to run the logic that handles the message. 
        # Since 'run' is an infinite loop, we can extract the logic or just run it in a thread 
        # and set running=False after a short delay.
        
        # Alternative: Just call the code block manually if I had extracted it, but I didn't.
        # So I will run it in a separate thread and stop it from main thread.
        
        def stop_orch():
            time.sleep(0.1)
            orch.running = False
            
        t = threading.Thread(target=stop_orch)
        t.start()
        
        orch.run()
        t.join()
        
        # VERIFICATION
        # Check if hset was called through RedisBus wrapper.
        self.assertTrue(mock_bus_instance.hset.called)
        args, _ = mock_bus_instance.hset.call_args
        self.assertEqual(args[0], 'workers:data')
        self.assertEqual(args[1], 'bench_worker')
        
        saved_data = json.loads(args[2])
        self.assertEqual(saved_data['worker_id'], 'bench_worker')
        self.assertEqual(saved_data['symbol'], 'ETH/USDT')
        self.assertIn('last_updated', saved_data)
        
        print("\nSUCCESS: Orchestrator correctly calls Redis HSET with worker data.")

if __name__ == "__main__":
    unittest.main()
