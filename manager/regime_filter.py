
import pandas as pd
import pandas_ta as ta
import logging
from workers.order_manager import OrderManager

class RegimeFilter:
    def __init__(self, config):
        self.logger = logging.getLogger("RegimeFilter")
        self.config = config
        
        # Thresholds
        regime_cfg = config.get('regime', {})
        self.adx_threshold = regime_cfg.get('adx_threshold', 30.0)
        self.timeframe = regime_cfg.get('timeframe', '1h')
        self.symbol = regime_cfg.get('symbol', 'SOL/USDT')
        
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

            # Get latest ADX value (ADX_14 or similar column)
            adx_col = next((c for c in adx_df.columns if str(c).startswith('ADX_')), None)
            if not adx_col:
                return 'UNKNOWN'
            current_adx = adx_df.iloc[-1][adx_col]
            if pd.isna(current_adx):
                return 'UNKNOWN'
            
            self.logger.info(f"Market Analysis [{self.symbol}]: ADX={current_adx:.2f}")

            if current_adx > self.adx_threshold:
                return 'TRENDING'
            else:
                return 'RANGING'

        except Exception as e:
            self.logger.error(f"Regime Analysis Failed: {e}")
            return 'ERROR'
