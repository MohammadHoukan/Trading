
import unittest
from unittest.mock import MagicMock, patch
import signal
import sys
import os
import threading
import time

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from workers.grid_bot import GridBot

class TestWorkerShutdown(unittest.TestCase):
    def setUp(self):
        # Prevent actual argparse parsing
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=MagicMock(pair='SOL/USDT', grids=20)):

            # Mock config to avoid Key Pool logic
            mock_config = {
                'exchange': {
                    'name': 'binance',
                    'mode': 'live',
                    'api_key': 'k', 
                    'secret': 's'
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'channels': {'status': 'status_chan', 'command': 'cmd_chan'}
                }
            }
            
            with patch('workers.grid_bot.load_config', return_value=mock_config):
                with patch('workers.grid_bot.get_redis_params', return_value={}):
                    with patch('workers.grid_bot.RedisBus'): # Mock Bus entirely
                        self.bot = GridBot('SOL/USDT', 20)
            
        self.bot.config = mock_config # Ensure it persists
        self.bot.bus = MagicMock()
        self.bot.logger = MagicMock()
        self.bot.order_manager = MagicMock()
        self.bot.db = MagicMock()
        self.bot.rate_limiter = MagicMock()
        self.bot.rate_limiter.acquire.return_value = True

    @patch('time.sleep') 
    def test_publish_stopped_on_loop_exit(self, mock_sleep):
        """Verify _publish_terminal_status('STOPPED') is called when run loop exits."""
        # Setup: run one iteration then stop
        self.bot.running = True
        
        # We need to break the loop. 
        # Strategy: mock _check_stale_data to set running=False side effect
        def side_effect():
            self.bot.running = False
            return False
            
        with patch.object(self.bot, '_check_stale_data', side_effect=side_effect):
            self.bot.run()
            
        # Verify publish called with STOPPED
        self.bot.bus.publish.assert_called()
        calls = self.bot.bus.publish.call_args_list
        # Found relevant status msg?
        found_stopped = False
        for call in calls:
            args, _ = call
            if args[0] == 'status_chan' and args[1].get('status') == 'STOPPED':
                found_stopped = True
                break
        self.assertTrue(found_stopped, "Did not find STOPPED status publication")

    def test_signal_handler_sets_running_false(self):
        """Verify SIGINT/SIGTERM handler sets self.running = False."""
        self.bot.running = True
        self.bot._handle_signal(signal.SIGINT, None)
        self.assertFalse(self.bot.running)

if __name__ == '__main__':
    unittest.main()
