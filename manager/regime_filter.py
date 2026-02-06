
import pandas as pd
import pandas_ta as ta
import logging
import sqlite3
import time
import os
from workers.order_manager import OrderManager

# ML imports (optional)
try:
    from ml.models.regime_classifier import RegimeClassifier
    from ml.features.regime_features import extract_regime_features
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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
        self.default_trending_scale = regime_cfg.get('trending_scale', 0.5)
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

        # ML classifier configuration
        smart_cfg = config.get('smart_features', {})
        ml_cfg = smart_cfg.get('regime_classifier', {})
        self.ml_enabled = smart_cfg.get('ml_enabled', False) and ml_cfg.get('enabled', False)
        self.ml_ensemble_weight = ml_cfg.get('ensemble_weight', 0.6)
        self.ml_confidence_threshold = ml_cfg.get('confidence_threshold', 0.7)
        self.ml_model_path = ml_cfg.get('model_path', 'data/models/regime_classifier.joblib')

        # Initialize ML classifier if enabled
        self.ml_classifier = None
        if self.ml_enabled and ML_AVAILABLE:
            self._init_ml_classifier()
        elif self.ml_enabled and not ML_AVAILABLE:
            self.logger.warning("ML features enabled but ml module not available")
            self.ml_enabled = False

        # Re-use OrderManager for data fetching
        self.data_source = OrderManager(
            config['exchange']['name'],
            config['exchange']['api_key'],
            config['exchange']['secret'],
            testnet=(config['exchange']['mode'] == 'testnet')
        )

    def _init_ml_classifier(self):
        """Initialize and load the ML classifier if available."""
        try:
            self.ml_classifier = RegimeClassifier(self.config)
            if os.path.exists(self.ml_model_path):
                if self.ml_classifier.load(self.ml_model_path):
                    self.logger.info(f"ML regime classifier loaded from {self.ml_model_path}")
                else:
                    self.logger.warning("Failed to load ML classifier, using rules only")
                    self.ml_enabled = False
            else:
                self.logger.info(f"ML model not found at {self.ml_model_path}, using rules only")
                self.ml_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize ML classifier: {e}")
            self.ml_enabled = False

    def get_thresholds(self, symbol):
        """Get thresholds for a specific symbol (with per-symbol overrides)."""
        thresholds = self.default_thresholds.copy()
        if symbol in self.per_symbol:
            overrides = self.per_symbol[symbol]
            for key in self.default_thresholds:
                if key in overrides:
                    thresholds[key] = overrides[key]
        return thresholds

    def get_trending_scale(self, symbol):
        """Get trending exposure scale for a symbol, with per-symbol override support."""
        scale = self.default_trending_scale
        if symbol in self.per_symbol:
            scale = self.per_symbol[symbol].get('trending_scale', scale)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 0.5
        # Clamp to sane range [0.05, 1.0] to avoid zero/negative order sizes.
        return max(0.05, min(1.0, scale))

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

            # Weighted average for rule-based score
            rule_score = 50.0
            if scores:
                total_weight = sum(self.weights[k] for k in scores.keys())
                rule_score = sum(scores[k] * self.weights[k] for k in scores.keys()) / total_weight

            # ===== ML Ensemble Scoring =====
            ml_score = None
            ml_confidence = 0.0
            if self.ml_enabled and self.ml_classifier:
                try:
                    ml_features = extract_regime_features(df)
                    ml_regime, ml_confidence = self.ml_classifier.predict(ml_features)
                    ml_score = self.ml_classifier.get_score(ml_features)

                    result['signals']['ml_regime'] = ml_regime
                    result['signals']['ml_confidence'] = round(ml_confidence, 3)
                    result['signals']['ml_score'] = round(ml_score, 1)
                except Exception as e:
                    self.logger.warning(f"ML prediction failed: {e}")

            # Combine rule-based and ML scores
            if ml_score is not None and ml_confidence >= self.ml_confidence_threshold:
                # Ensemble: weighted combination
                final_score = (
                    self.ml_ensemble_weight * ml_score +
                    (1 - self.ml_ensemble_weight) * rule_score
                )
                result['signals']['ensemble_method'] = 'weighted'
            else:
                # Fallback to rule-based only
                final_score = rule_score
                if ml_score is not None:
                    result['signals']['ensemble_method'] = 'rules_only_low_confidence'
                else:
                    result['signals']['ensemble_method'] = 'rules_only'

            result['score'] = round(final_score, 1)
            
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
                result['scale'] = self.get_trending_scale(symbol)
            
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
