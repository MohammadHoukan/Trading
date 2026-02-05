
import pandas as pd
import pandas_ta as ta
import logging
from workers.order_manager import OrderManager

class RegimeFilter:
    def __init__(self, config):
        self.logger = logging.getLogger("RegimeFilter")
        self.config = config
        
        # Thresholds
        self.adx_threshold = config.get('regime', {}).get('adx_threshold', 30.0)
        self.timeframe = config.get('regime', {}).get('timeframe', '1h')
        self.symbol = "SOL/USDT" # Monitor the main pair for now, or configurable
        
        # Re-use OrderManager for data fetching
        self.data_source = OrderManager(
            config['exchange']['name'],
            config['exchange']['api_key'],
            config['exchange']['secret'],
            testnet=(config['exchange']['mode'] == 'testnet')
        )

    def analyze_market(self):
        """
        Fetch candles and determine regime.
        Returns: 'TRENDING' (Bad for grid) or 'RANGING' (Good for grid)
        """
        try:
            # Fetch OHLCV
            candles = self.data_source.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
            if not candles:
                self.logger.warning("No candle data fetched.")
                return 'UNKNOWN'

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate ADX
            # pandas_ta requires dataframe with high, low, close
            adx_df = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)
            
            if adx_df is None or adx_df.empty:
                return 'UNKNOWN'

            # Get latest ADX value (ADX_14 column)
            current_adx = adx_df.iloc[-1]['ADX_14']
            
            self.logger.info(f"Market Analysis [{self.symbol}]: ADX={current_adx:.2f}")

            if current_adx > self.adx_threshold:
                return 'TRENDING'
            else:
                return 'RANGING'

        except Exception as e:
            self.logger.error(f"Regime Analysis Failed: {e}")
            return 'ERROR'
