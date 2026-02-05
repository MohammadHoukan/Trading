
import pandas as pd
import pandas_ta as ta
import logging
import sqlite3
import time
from workers.order_manager import OrderManager

class RegimeFilter:
    """
    Enhanced Regime Filter with multiple signals:
    - ADX (trend strength)
    - ATR/Price (volatility)
    - MA Distance (trend direction)
    - Fill Rate (execution quality from DB)
    
    Combines into weighted composite score for regime decision.
    """
    
    def __init__(self, config, db_path='swarm.db'):
        self.logger = logging.getLogger("RegimeFilter")
        self.config = config
        self.db_path = db_path
        
        # Default thresholds
        regime_cfg = config.get('regime', {})
        self.default_thresholds = {
            'adx_threshold': regime_cfg.get('adx_threshold', 30.0),
            'volatility_threshold': regime_cfg.get('volatility_threshold', 0.03),  # 3% ATR/price
            'ma_distance_threshold': regime_cfg.get('ma_distance_threshold', 0.02),  # 2% from MA
            'fill_rate_threshold': regime_cfg.get('fill_rate_threshold', 0.3),  # 30% minimum
        }
        self.timeframe = regime_cfg.get('timeframe', '1h')
        self.default_symbol = regime_cfg.get('symbol', 'SOL/USDT')
        
        # Per-symbol overrides
        self.per_symbol = regime_cfg.get('per_symbol', {})
        
        # Signal weights for composite score
        self.weights = {
            'adx': 0.35,
            'volatility': 0.25,
            'ma_distance': 0.25,
            'fill_rate': 0.15,
        }
        
        # Re-use OrderManager for data fetching
        self.data_source = OrderManager(
            config['exchange']['name'],
            config['exchange']['api_key'],
            config['exchange']['secret'],
            testnet=(config['exchange']['mode'] == 'testnet')
        )

    def get_thresholds(self, symbol):
        """Get thresholds for a specific symbol (with per-symbol overrides)."""
        thresholds = self.default_thresholds.copy()
        if symbol in self.per_symbol:
            thresholds.update(self.per_symbol[symbol])
        return thresholds

    def analyze_market(self, symbol=None):
        """
        Fetch candles and determine regime with detailed signals.
        
        Args:
            symbol: Trading pair. If None, uses default_symbol.
            
        Returns:
            dict with keys:
                - regime: 'TRENDING' | 'RANGING' | 'UNKNOWN' | 'ERROR'
                - score: 0-100 composite score (higher = more favorable for grid)
                - signals: dict of individual signal values
        """
        symbol = symbol or self.default_symbol
        thresholds = self.get_thresholds(symbol)
        
        result = {
            'symbol': symbol,
            'regime': 'UNKNOWN',
            'score': 50,
            'signals': {},
            'recommendation': 'HOLD',
            'scale': 1.0,
        }
        
        try:
            # Fetch OHLCV
            candles = self.data_source.exchange.fetch_ohlcv(symbol, self.timeframe, limit=100)
            if not candles:
                self.logger.warning(f"No candle data for {symbol}")
                return result

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # ===== Signal 1: ADX (Trend Strength) =====
            adx_df = df.ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)
            adx_col = next((c for c in adx_df.columns if str(c).startswith('ADX_')), None)
            current_adx = adx_df.iloc[-1][adx_col] if adx_col else None
            
            # ===== Signal 2: ATR/Price (Volatility) =====
            atr = df.ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            current_price = df['close'].iloc[-1]
            volatility = (atr.iloc[-1] / current_price) if atr is not None and not atr.empty else None
            
            # ===== Signal 3: MA Distance =====
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            ma_distance = (current_price - sma_50) / sma_50 if sma_50 else None
            
            # ===== Signal 4: Fill Rate (from DB) =====
            fill_rate = self._get_fill_rate(symbol)
            
            # Store signals
            result['signals'] = {
                'adx': round(current_adx, 2) if current_adx and not pd.isna(current_adx) else None,
                'volatility': round(volatility, 4) if volatility and not pd.isna(volatility) else None,
                'ma_distance': round(ma_distance, 4) if ma_distance and not pd.isna(ma_distance) else None,
                'fill_rate': round(fill_rate, 4) if fill_rate else None,
                'price': round(current_price, 2),
                'sma_50': round(sma_50, 2) if sma_50 and not pd.isna(sma_50) else None,
            }
            
            # ===== Calculate Composite Score =====
            # Each signal contributes 0-100 points, weighted
            scores = {}
            
            # ADX: Lower is better for grid (ranging = good)
            if current_adx is not None and not pd.isna(current_adx):
                adx_score = max(0, 100 - (current_adx / thresholds['adx_threshold']) * 50)
                scores['adx'] = min(100, adx_score)
            
            # Volatility: Moderate is best (too low = no profit, too high = risk)
            if volatility is not None and not pd.isna(volatility):
                vol_ratio = volatility / thresholds['volatility_threshold']
                if vol_ratio < 0.5:
                    vol_score = 50  # Too low volatility
                elif vol_ratio < 1.5:
                    vol_score = 100  # Sweet spot
                else:
                    vol_score = max(0, 100 - (vol_ratio - 1.5) * 50)  # Too high
                scores['volatility'] = vol_score
            
            # MA Distance: Closer to MA = better for grid
            if ma_distance is not None and not pd.isna(ma_distance):
                abs_dist = abs(ma_distance)
                ma_score = max(0, 100 - (abs_dist / thresholds['ma_distance_threshold']) * 50)
                scores['ma_distance'] = min(100, ma_score)
            
            # Fill Rate: Higher is better
            if fill_rate is not None:
                fill_score = min(100, (fill_rate / thresholds['fill_rate_threshold']) * 100)
                scores['fill_rate'] = fill_score
            
            # Weighted average
            if scores:
                total_weight = sum(self.weights[k] for k in scores.keys())
                composite = sum(scores[k] * self.weights[k] for k in scores.keys()) / total_weight
                result['score'] = round(composite, 1)
            
            # ===== Determine Regime =====
            if result['score'] >= 60:
                result['regime'] = 'RANGING'
                result['recommendation'] = 'RUN'
                result['scale'] = 1.0
            elif result['score'] >= 40:
                result['regime'] = 'UNCERTAIN'
                result['recommendation'] = 'HOLD'  # Keep current state
                result['scale'] = 1.0
            else:
                result['regime'] = 'TRENDING'
                result['recommendation'] = 'REDUCE_EXPOSURE'
                trending_scale = self.config.get('regime', {}).get('trending_scale', 0.5)
                try:
                    trending_scale = float(trending_scale)
                except (TypeError, ValueError):
                    trending_scale = 0.5
                # Clamp to sane range [0.05, 1.0] to avoid zero/negative order sizes.
                result['scale'] = max(0.05, min(1.0, trending_scale))
            
            self.logger.info(
                f"Regime [{symbol}]: {result['regime']} (score={result['score']}) "
                f"ADX={result['signals'].get('adx')} VOL={result['signals'].get('volatility')} "
                f"MA_DIST={result['signals'].get('ma_distance')} FILL={result['signals'].get('fill_rate')}"
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Regime Analysis Failed for {symbol}: {e}")
            result['regime'] = 'ERROR'
            return result

    def _get_fill_rate(self, symbol, lookback_hours=24):
        """Get fill rate from grid_events database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = time.time() - (lookback_hours * 3600)
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN event_type = 'FILL' THEN 1 ELSE 0 END) as fills,
                    SUM(CASE WHEN event_type = 'PLACE' THEN 1 ELSE 0 END) as places
                FROM grid_events 
                WHERE symbol = ? AND timestamp > ?
            ''', (symbol, since))
            
            row = cursor.fetchone()
            
            if row and row[1] and row[1] > 0:
                return row[0] / row[1]
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to fetch fill rate: {e}")
            return None
        finally:
            if conn is not None:
                conn.close()
