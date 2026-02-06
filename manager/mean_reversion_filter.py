"""
Mean Reversion Entry Filter.

Uses Z-score analysis to identify optimal entry/exit timing based on
price deviation from the moving average.

Z-score = (price - SMA) / std_dev

Entry signals when price is extended from mean (high |z-score|)
Exit signals when price returns to mean (low |z-score|)
"""

import logging
import pandas as pd
from typing import Dict, Optional
from workers.order_manager import OrderManager


class MeanReversionFilter:
    """
    Mean reversion filter for grid trading entry timing.

    High |z-score| indicates price is extended - good time to enter grid
    as mean reversion is more likely.

    Low |z-score| indicates price is near mean - may want to wait for
    better entry or reduce exposure.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger("MeanReversionFilter")
        self.config = config

        # Load configuration
        mr_cfg = config.get('smart_features', {}).get('mean_reversion', {})
        self.enabled = mr_cfg.get('enabled', True)
        self.sma_period = mr_cfg.get('sma_period', 20)
        self.std_period = mr_cfg.get('std_period', 50)
        self.entry_z_threshold = mr_cfg.get('entry_z_threshold', 1.5)
        self.exit_z_threshold = mr_cfg.get('exit_z_threshold', 0.5)

        # Timeframe for analysis
        self.timeframe = config.get('regime', {}).get('timeframe', '1h')

        # Data source for fetching candles
        self.data_source = None
        self._init_data_source()

    def _init_data_source(self):
        """Initialize data source for fetching candle data."""
        try:
            exchange_cfg = self.config.get('exchange', {})
            self.data_source = OrderManager(
                exchange_cfg.get('name', 'binance'),
                exchange_cfg.get('api_key', ''),
                exchange_cfg.get('secret', ''),
                testnet=(exchange_cfg.get('mode') == 'testnet')
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize data source: {e}")

    def analyze(self, symbol: str) -> Dict:
        """
        Analyze mean reversion status for a symbol.

        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')

        Returns:
            dict with keys:
                - action: 'ENTER' | 'HOLD' | 'EXIT'
                - z_score: current z-score value
                - price: current price
                - sma: simple moving average
                - std: standard deviation
                - recommendation: human-readable recommendation
        """
        result = {
            'symbol': symbol,
            'action': 'HOLD',
            'z_score': 0.0,
            'price': None,
            'sma': None,
            'std': None,
            'recommendation': 'Insufficient data',
        }

        if not self.enabled:
            result['recommendation'] = 'Mean reversion filter disabled'
            return result

        if not self.data_source:
            self.logger.warning("No data source available")
            result['recommendation'] = 'No data source'
            return result

        try:
            # Fetch candles - need enough for both SMA and std calculation
            limit = max(self.sma_period, self.std_period) + 10
            candles = self.data_source.exchange.fetch_ohlcv(
                symbol, self.timeframe, limit=limit
            )

            if not candles or len(candles) < limit:
                self.logger.warning(f"Insufficient candle data for {symbol}")
                result['recommendation'] = 'Insufficient data'
                return result

            df = pd.DataFrame(
                candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Calculate indicators
            current_price = df['close'].iloc[-1]
            sma = df['close'].rolling(self.sma_period).mean().iloc[-1]
            std = df['close'].rolling(self.std_period).std().iloc[-1]

            if pd.isna(sma) or pd.isna(std) or std == 0:
                result['recommendation'] = 'Invalid indicator values'
                return result

            # Calculate Z-score
            z_score = (current_price - sma) / std

            result['price'] = round(current_price, 6)
            result['sma'] = round(sma, 6)
            result['std'] = round(std, 6)
            result['z_score'] = round(z_score, 4)

            # Determine action based on Z-score
            abs_z = abs(z_score)

            if abs_z >= self.entry_z_threshold:
                result['action'] = 'ENTER'
                direction = 'above' if z_score > 0 else 'below'
                result['recommendation'] = (
                    f"Price extended {direction} mean (z={z_score:.2f}). "
                    f"Good entry for mean reversion."
                )
            elif abs_z <= self.exit_z_threshold:
                result['action'] = 'EXIT'
                result['recommendation'] = (
                    f"Price near mean (z={z_score:.2f}). "
                    f"Consider reducing exposure or waiting."
                )
            else:
                result['action'] = 'HOLD'
                result['recommendation'] = (
                    f"Price moderately extended (z={z_score:.2f}). "
                    f"Hold current position."
                )

            self.logger.info(
                f"MeanReversion [{symbol}]: z={z_score:.2f}, action={result['action']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Mean reversion analysis failed for {symbol}: {e}")
            result['recommendation'] = f'Analysis error: {e}'
            return result

    def should_enter(self, symbol: str) -> bool:
        """
        Simple check if conditions favor entering a grid position.

        Args:
            symbol: Trading pair

        Returns:
            True if mean reversion conditions favor entry
        """
        if not self.enabled:
            return True  # Don't block if disabled

        analysis = self.analyze(symbol)
        return analysis['action'] == 'ENTER'

    def should_exit(self, symbol: str) -> bool:
        """
        Simple check if conditions suggest exiting/reducing position.

        Args:
            symbol: Trading pair

        Returns:
            True if price has reverted to mean
        """
        if not self.enabled:
            return False  # Don't suggest exit if disabled

        analysis = self.analyze(symbol)
        return analysis['action'] == 'EXIT'

    def get_z_score(self, symbol: str) -> Optional[float]:
        """
        Get just the Z-score for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Z-score value or None if unavailable
        """
        analysis = self.analyze(symbol)
        z = analysis.get('z_score')
        return z if z != 0.0 or analysis.get('price') is not None else None
