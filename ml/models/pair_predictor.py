"""
Gradient Boosting Pair Profitability Predictor.

Predicts the expected profitability of trading a specific pair
based on market characteristics and historical performance metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, ML features disabled")

from ml.persistence.model_store import save_model, load_model
from ml.features.pair_features import extract_pair_features, get_feature_names


class PairPredictor:
    """
    Gradient Boosting regressor for pair profitability prediction.

    Outputs a continuous profitability score that can be used
    for pair ranking and selection.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the predictor.

        Args:
            config: Optional configuration dict with model hyperparameters
        """
        self.logger = logging.getLogger("PairPredictor")
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_names = get_feature_names()
        self.is_trained = False

        # Default hyperparameters (can be overridden by config)
        ml_config = self.config.get('pair_predictor', {})
        self.n_estimators = ml_config.get('n_estimators', 100)
        self.learning_rate = ml_config.get('learning_rate', 0.1)
        self.max_depth = ml_config.get('max_depth', 5)
        self.min_samples_split = ml_config.get('min_samples_split', 5)
        self.min_samples_leaf = ml_config.get('min_samples_leaf', 2)
        self.subsample = ml_config.get('subsample', 0.8)
        self.random_state = ml_config.get('random_state', 42)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the predictor on labeled data.

        Args:
            X: Feature DataFrame
            y: Target Series (continuous profitability score)
            test_size: Fraction of data to use for testing

        Returns:
            Dict with training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        self.logger.info(f"Training pair predictor on {len(X)} samples")

        # Handle missing values
        X = X.fillna(X.median())

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )

        # Initialize model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = dict(zip(
            X.columns if hasattr(X, 'columns') else self.feature_names,
            self.model.feature_importances_
        ))

        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance,
        }

        self.logger.info(f"Training complete. R2: {r2:.3f}, RMSE: {np.sqrt(mse):.4f}")
        return metrics

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict profitability score for a single pair.

        Args:
            features: Dict of feature name -> value

        Returns:
            Profitability score (higher = better)
        """
        if not self.is_trained or self.model is None:
            self.logger.warning("Model not trained, returning neutral score")
            return 0.0

        try:
            # Create feature vector in correct order
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])

            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)

            # Scale
            if self.scaler:
                X = self.scaler.transform(X)

            # Predict
            score = self.model.predict(X)[0]
            return float(score)

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0.0

    def predict_from_data(
        self,
        market_data: Dict,
        backtest_metrics: Optional[Dict] = None
    ) -> float:
        """
        Predict profitability directly from market data.

        Args:
            market_data: Dict with market characteristics
            backtest_metrics: Optional dict with historical performance

        Returns:
            Profitability score
        """
        try:
            features = extract_pair_features(market_data, backtest_metrics)
            return self.predict(features)
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return 0.0

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict profitability for multiple pairs.

        Args:
            features_df: DataFrame with features for multiple pairs

        Returns:
            Array of profitability scores
        """
        if not self.is_trained or self.model is None:
            return np.zeros(len(features_df))

        try:
            # Ensure correct feature order
            X = features_df[self.feature_names].values

            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)

            # Scale
            if self.scaler:
                X = self.scaler.transform(X)

            return self.model.predict(X)

        except Exception as e:
            self.logger.error(f"Batch prediction error: {e}")
            return np.zeros(len(features_df))

    def get_normalized_score(self, features: Dict[str, float], scale: float = 100.0) -> float:
        """
        Get a normalized score (0-100) for pair ranking.

        Uses sigmoid transformation to bound the output.

        Args:
            features: Dict of feature values
            scale: Output scale (default 100)

        Returns:
            Normalized score from 0 to scale
        """
        raw_score = self.predict(features)

        # Sigmoid transformation to bound output
        # Assumes raw scores are roughly in range [-0.1, 0.1]
        normalized = 1 / (1 + np.exp(-raw_score * 10))

        return normalized * scale

    def save(self, path: str) -> bool:
        """
        Save the trained model to disk.

        Args:
            path: File path (should end in .joblib)

        Returns:
            True if successful
        """
        if not self.is_trained:
            self.logger.warning("Cannot save untrained model")
            return False

        metadata = {
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
        }

        # Package model and scaler together
        package = {
            'model': self.model,
            'scaler': self.scaler,
        }

        return save_model(package, path, metadata)

    def load(self, path: str) -> bool:
        """
        Load a trained model from disk.

        Args:
            path: File path

        Returns:
            True if successful
        """
        package = load_model(path)
        if package is None:
            return False

        try:
            if isinstance(package, dict):
                self.model = package.get('model')
                self.scaler = package.get('scaler')
            else:
                # Legacy format
                self.model = package
                self.scaler = None

            self.is_trained = self.model is not None
            self.logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    @property
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return None

        return dict(zip(self.feature_names, self.model.feature_importances_))
