import sys
import os
import time
import threading
import logging

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from workers.grid_bot import GridBot
from shared.messaging import RedisBus
from shared.config import get_redis_params

logging.basicConfig(level=logging.INFO)

def mock_worker(symbol):
    try:
        # We Initialize GridBot which triggers the locking logic in __init__
        # We don't run run() to avoid hitting the exchange
        bot = GridBot(symbol, 10, config_path='config/settings.yaml')
        print(f"[{symbol}] SUCCESS: Locked Key {bot.api_key[-4:]} with Lock ID {bot.key_lock_id}")
        
        # Keep alive for a bit to test concurrency
        time.sleep(5)
        bot.running = False # Stop renew thread
    except Exception as e:
        print(f"[{symbol}] FAILED: {e}")

if __name__ == "__main__":
    # Ensure Redis is clean of locks for test
    # (In real life, previous crashes might leave locks for 60s)
    
    # 1. Start Two Workers concurrently
    t1 = threading.Thread(target=mock_worker, args=("SOL/USDT",))
    t2 = threading.Thread(target=mock_worker, args=("ETH/USDT",))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
