"""
Random Forest Regime Classifier.

Predicts whether market conditions are FAVORABLE or UNFAVORABLE
for grid trading based on technical indicators and market structure.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, ML features disabled")

from ml.persistence.model_store import save_model, load_model
from ml.features.regime_features import extract_regime_features, get_feature_names


class RegimeClassifier:
    """
    Random Forest classifier for market regime prediction.

    Predicts FAVORABLE (1) or UNFAVORABLE (0) for grid trading.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the classifier.

        Args:
            config: Optional configuration dict with model hyperparameters
        """
        self.logger = logging.getLogger("RegimeClassifier")
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_names = get_feature_names()
        self.is_trained = False

        # Default hyperparameters (can be overridden by config)
        ml_config = self.config.get('regime_classifier', {})
        self.n_estimators = ml_config.get('n_estimators', 100)
        self.max_depth = ml_config.get('max_depth', 10)
        self.min_samples_split = ml_config.get('min_samples_split', 5)
        self.min_samples_leaf = ml_config.get('min_samples_leaf', 2)
        self.class_weight = ml_config.get('class_weight', 'balanced')
        self.random_state = ml_config.get('random_state', 42)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train the classifier on labeled data.

        Args:
            X: Feature DataFrame
            y: Label Series (1 = FAVORABLE, 0 = UNFAVORABLE)
            test_size: Fraction of data to use for testing

        Returns:
            Dict with training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        self.logger.info(f"Training regime classifier on {len(X)} samples")

        # Handle missing values
        X = X.fillna(X.median())

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        feature_importance = dict(zip(
            X.columns if hasattr(X, 'columns') else self.feature_names,
            self.model.feature_importances_
        ))

        metrics = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }

        self.logger.info(f"Training complete. Accuracy: {accuracy:.3f}")
        return metrics

    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Predict regime for a single observation.

        Args:
            features: Dict of feature name -> value

        Returns:
            Tuple of (regime string, confidence score)
            regime: 'FAVORABLE' or 'UNFAVORABLE'
            confidence: 0.0 to 1.0
        """
        if not self.is_trained or self.model is None:
            self.logger.warning("Model not trained, returning default")
            return 'UNKNOWN', 0.0

        try:
            # Create feature vector in correct order
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])

            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)

            # Scale
            if self.scaler:
                X = self.scaler.transform(X)

            # Predict with probabilities
            proba = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]

            # Get confidence (probability of predicted class)
            confidence = max(proba)

            regime = 'FAVORABLE' if prediction == 1 else 'UNFAVORABLE'
            return regime, confidence

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 'UNKNOWN', 0.0

    def predict_from_candles(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict regime directly from OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (regime, confidence)
        """
        try:
            features = extract_regime_features(df)
            return self.predict(features)
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return 'UNKNOWN', 0.0

    def get_score(self, features: Dict[str, float]) -> float:
        """
        Get a continuous score (0-100) for regime favorability.

        Args:
            features: Dict of feature values

        Returns:
            Score from 0 (unfavorable) to 100 (favorable)
        """
        if not self.is_trained or self.model is None:
            return 50.0  # Neutral if not trained

        try:
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
            X = np.nan_to_num(X, nan=0.0)

            if self.scaler:
                X = self.scaler.transform(X)

            proba = self.model.predict_proba(X)[0]
            # Index 1 is probability of FAVORABLE class
            favorable_prob = proba[1] if len(proba) > 1 else proba[0]

            return favorable_prob * 100

        except Exception as e:
            self.logger.error(f"Score calculation error: {e}")
            return 50.0

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
