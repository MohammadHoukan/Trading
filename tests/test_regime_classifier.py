"""Tests for the ML Regime Classifier."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os

# Skip tests if scikit-learn not available
sklearn_available = True
try:
    import sklearn
except ImportError:
    sklearn_available = False


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not installed")
class TestRegimeClassifier:
    """Test suite for RegimeClassifier."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data for testing."""
        np.random.seed(42)
        n = 200

        return pd.DataFrame({
            'adx': np.random.uniform(10, 50, n),
            'atr_pct': np.random.uniform(0.01, 0.05, n),
            'ma_distance_20': np.random.uniform(-0.05, 0.05, n),
            'ma_distance_50': np.random.uniform(-0.10, 0.10, n),
            'rsi': np.random.uniform(20, 80, n),
            'macd_histogram_pct': np.random.uniform(-0.01, 0.01, n),
            'volume_ratio': np.random.uniform(0.5, 2.0, n),
            'bb_width': np.random.uniform(0.02, 0.08, n),
            'price_position': np.random.uniform(0, 1, n),
            'hour_of_day': np.random.randint(0, 24, n),
            'day_of_week': np.random.randint(0, 7, n),
            'return_1h': np.random.uniform(-0.02, 0.02, n),
            'return_4h': np.random.uniform(-0.05, 0.05, n),
            'return_24h': np.random.uniform(-0.10, 0.10, n),
        })

    @pytest.fixture
    def sample_labels(self, sample_features):
        """Create sample labels based on features."""
        # Simple rule: favorable when ADX < 25 and volatility moderate
        adx = sample_features['adx']
        vol = sample_features['atr_pct']
        labels = ((adx < 30) & (vol > 0.015) & (vol < 0.04)).astype(int)
        return pd.Series(labels)

    def test_init(self):
        """Test classifier initialization."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()

        assert clf.model is None
        assert clf.is_trained is False
        assert clf.n_estimators == 100
        assert clf.max_depth == 10

    def test_init_with_config(self):
        """Test classifier initialization with custom config."""
        from ml.models.regime_classifier import RegimeClassifier

        config = {
            'regime_classifier': {
                'n_estimators': 50,
                'max_depth': 5,
            }
        }
        clf = RegimeClassifier(config)

        assert clf.n_estimators == 50
        assert clf.max_depth == 5

    def test_train(self, sample_features, sample_labels):
        """Test training the classifier."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        metrics = clf.train(sample_features, sample_labels)

        assert clf.is_trained is True
        assert clf.model is not None
        assert 'accuracy' in metrics
        assert 'feature_importance' in metrics
        assert metrics['accuracy'] > 0.5  # Should be better than random

    def test_predict(self, sample_features, sample_labels):
        """Test prediction on single observation."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        clf.train(sample_features, sample_labels)

        # Create a test observation
        features = sample_features.iloc[0].to_dict()
        regime, confidence = clf.predict(features)

        assert regime in ['FAVORABLE', 'UNFAVORABLE', 'UNKNOWN']
        assert 0 <= confidence <= 1

    def test_predict_untrained(self):
        """Test prediction without training returns unknown."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        features = {'adx': 25, 'atr_pct': 0.02}
        regime, confidence = clf.predict(features)

        assert regime == 'UNKNOWN'
        assert confidence == 0.0

    def test_get_score(self, sample_features, sample_labels):
        """Test getting continuous score."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        clf.train(sample_features, sample_labels)

        features = sample_features.iloc[0].to_dict()
        score = clf.get_score(features)

        assert 0 <= score <= 100

    def test_save_and_load(self, sample_features, sample_labels, tmp_path):
        """Test model persistence."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        clf.train(sample_features, sample_labels)

        # Save
        model_path = str(tmp_path / "test_model.joblib")
        assert clf.save(model_path) is True
        assert os.path.exists(model_path)

        # Load into new classifier
        clf2 = RegimeClassifier()
        assert clf2.load(model_path) is True
        assert clf2.is_trained is True

        # Verify predictions match
        features = sample_features.iloc[0].to_dict()
        regime1, conf1 = clf.predict(features)
        regime2, conf2 = clf2.predict(features)
        assert regime1 == regime2
        assert abs(conf1 - conf2) < 0.01

    def test_feature_importance(self, sample_features, sample_labels):
        """Test feature importance retrieval."""
        from ml.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier()
        clf.train(sample_features, sample_labels)

        importance = clf.feature_importance
        assert importance is not None
        assert len(importance) > 0
        assert sum(importance.values()) > 0.99  # Should sum to ~1


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not installed")
class TestRegimeFeatureExtraction:
    """Test suite for regime feature extraction."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        timestamps = list(range(0, n * 3600000, 3600000))
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, n),
        })

    def test_extract_regime_features(self, sample_ohlcv):
        """Test feature extraction from OHLCV data."""
        from ml.features.regime_features import extract_regime_features

        features = extract_regime_features(sample_ohlcv)

        assert 'adx' in features
        assert 'atr_pct' in features
        assert 'rsi' in features
        assert 'volume_ratio' in features

    def test_extract_insufficient_data(self):
        """Test feature extraction with insufficient data."""
        from ml.features.regime_features import extract_regime_features

        df = pd.DataFrame({
            'timestamp': [0],
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [1000],
        })

        with pytest.raises(ValueError, match="Need at least"):
            extract_regime_features(df)
