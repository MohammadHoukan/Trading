
import unittest
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil
from unittest.mock import patch, MagicMock
from backtest.data_fetcher import load_cached, _cache_path, CACHE_DIR

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        # Ensure test cache dir exists
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        # Create a dummy cache file for SOL_USDT_1h
        self.symbol = 'SOL/USDT'
        self.timeframe = '1h'
        self.path = _cache_path(self.symbol, self.timeframe)
        
        # Create data spanning 20 days ago to now
        dates = pd.date_range(end=datetime.utcnow(), periods=20*24, freq='1h')
        df = pd.DataFrame(index=dates, data={'close': [100]*len(dates)})
        df.index.name = 'timestamp' # FIX: Explicitly name index so to_csv includes header
        df.to_csv(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_load_cached_miss_short_duration(self):
        """Request 30 days, cache only has 20 days. Should return None (miss)."""
        # We request 30 days
        # load_cached checks if start index is old enough.
        # Cache starts T-20. We need T-30.
        result = load_cached(self.symbol, self.timeframe, days=30)
        self.assertIsNone(result, "Should match miss because cache is too short")

    def test_load_cached_hit_sufficient_duration(self):
        """Request 10 days, cache has 20 days. Should return DataFrame (hit)."""
        result = load_cached(self.symbol, self.timeframe, days=10)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 20*24) # Returns full cached file

if __name__ == '__main__':
    unittest.main()
