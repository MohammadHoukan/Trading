"""Tests for the ML-enhanced pair scoring system."""

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
class TestPairPredictor:
    """Test suite for PairPredictor."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data for testing."""
        np.random.seed(42)
        n = 100

        return pd.DataFrame({
            'volume_24h_log': np.random.uniform(5, 9, n),
            'spread_pct': np.random.uniform(0.05, 0.5, n),
            'maker_fee': np.random.uniform(0.05, 0.15, n),
            'taker_fee': np.random.uniform(0.05, 0.15, n),
            'avg_fee': np.random.uniform(0.05, 0.15, n),
            'spread_fee_ratio': np.random.uniform(0.5, 5, n),
            'sharpe_ratio': np.random.uniform(-1, 3, n),
            'win_rate': np.random.uniform(0.3, 0.7, n),
            'max_drawdown': np.random.uniform(0, 0.15, n),
            'profit_factor': np.random.uniform(0.5, 3, n),
            'avg_trade_return': np.random.uniform(-0.01, 0.02, n),
            'trade_count_log': np.random.uniform(1, 3, n),
            'consistency_score': np.random.uniform(0, 1.5, n),
            'risk_adj_profit': np.random.uniform(0.5, 3, n),
        })

    @pytest.fixture
    def sample_labels(self, sample_features):
        """Create sample profitability labels."""
        # Simple formula: higher sharpe + win_rate = higher profitability
        labels = (
            0.5 * sample_features['sharpe_ratio'] +
            0.3 * sample_features['win_rate'] -
            0.2 * sample_features['max_drawdown']
        ) / 100 + np.random.randn(len(sample_features)) * 0.001
        return pd.Series(labels)

    def test_init(self):
        """Test predictor initialization."""
        from ml.models.pair_predictor import PairPredictor

        pred = PairPredictor()

        assert pred.model is None
        assert pred.is_trained is False
        assert pred.n_estimators == 100
        assert pred.learning_rate == 0.1

    def test_train(self, sample_features, sample_labels):
        """Test training the predictor."""
        from ml.models.pair_predictor import PairPredictor

        pred = PairPredictor()
        metrics = pred.train(sample_features, sample_labels)

        assert pred.is_trained is True
        assert pred.model is not None
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'feature_importance' in metrics

    def test_predict(self, sample_features, sample_labels):
        """Test prediction on single observation."""
        from ml.models.pair_predictor import PairPredictor

        pred = PairPredictor()
        pred.train(sample_features, sample_labels)

        features = sample_features.iloc[0].to_dict()
        score = pred.predict(features)

        assert isinstance(score, float)

    def test_predict_untrained(self):
        """Test prediction without training returns zero."""
        from ml.models.pair_predictor import PairPredictor

        pred = PairPredictor()
        features = {'volume_24h_log': 7, 'spread_pct': 0.1}
        score = pred.predict(features)

        assert score == 0.0

    def test_get_normalized_score(self, sample_features, sample_labels):
        """Test getting normalized score."""
        from ml.models.pair_predictor import PairPredictor

        pred = PairPredictor()
        pred.train(sample_features, sample_labels)

        features = sample_features.iloc[0].to_dict()
        score = pred.get_normalized_score(features)

        assert 0 <= score <= 100

    def test_save_and_load(self, sample_features, sample_labels, tmp_path):
        """Test model persistence."""
        from ml.models.pair_predictor import PairPredictor

        pred = PairPredictor()
        pred.train(sample_features, sample_labels)

        # Save
        model_path = str(tmp_path / "test_pair_model.joblib")
        assert pred.save(model_path) is True
        assert os.path.exists(model_path)

        # Load into new predictor
        pred2 = PairPredictor()
        assert pred2.load(model_path) is True
        assert pred2.is_trained is True

        # Verify predictions match
        features = sample_features.iloc[0].to_dict()
        score1 = pred.predict(features)
        score2 = pred2.predict(features)
        assert abs(score1 - score2) < 0.01


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not installed")
class TestPairFeatureExtraction:
    """Test suite for pair feature extraction."""

    def test_extract_pair_features_market_only(self):
        """Test feature extraction with market data only."""
        from ml.features.pair_features import extract_pair_features

        market_data = {
            'volume_24h': 5_000_000,
            'spread_pct': 0.1,
            'maker_fee': 0.1,
            'taker_fee': 0.1,
        }

        features = extract_pair_features(market_data)

        assert 'volume_24h_log' in features
        assert 'spread_pct' in features
        assert 'spread_fee_ratio' in features
        # Backtest features should be neutral
        assert features['sharpe_ratio'] == 0.0
        assert features['win_rate'] == 0.5

    def test_extract_pair_features_with_backtest(self):
        """Test feature extraction with backtest metrics."""
        from ml.features.pair_features import extract_pair_features

        market_data = {
            'volume_24h': 5_000_000,
            'spread_pct': 0.1,
            'maker_fee': 0.1,
            'taker_fee': 0.1,
        }
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': 0.05,
            'profit_factor': 2.0,
            'avg_trade_return': 0.005,
            'trade_count': 100,
        }

        features = extract_pair_features(market_data, backtest_metrics)

        assert features['sharpe_ratio'] == 1.5
        assert features['win_rate'] == 0.6
        assert features['profit_factor'] == 2.0


class TestEnhancedPairScorer:
    """Test suite for enhanced pair scoring with ML."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        return {
            'exchange': {
                'name': 'binance',
                'api_key': 'test_key',
                'secret': 'test_secret',
                'mode': 'testnet',
            },
            'pair_scorer': {
                'weight_volume': 0.15,
                'weight_spread': 0.20,
                'weight_fees': 0.10,
                'weight_fill_rate': 0.15,
                'weight_backtest': 0.40,
            },
            'smart_features': {
                'ml_enabled': False,  # Disable ML for basic tests
                'pair_ranking_ml': {
                    'enabled': False,
                    'ml_weight': 0.4,
                }
            }
        }

    @patch('manager.pair_scorer.OrderManager')
    @patch('manager.pair_scorer.Database')
    def test_score_backtest(self, mock_db, mock_om, mock_config):
        """Test backtest metrics scoring."""
        from manager.pair_scorer import PairScorer

        mock_om_instance = MagicMock()
        mock_om.return_value = mock_om_instance

        mock_db_instance = MagicMock()
        mock_db_instance.get_backtest_metrics.return_value = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': 0.03,
            'profit_factor': 2.0,
            'trade_count': 50,
        }
        mock_db.return_value = mock_db_instance

        scorer = PairScorer(mock_config)

        # Test scoring with good metrics
        metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': 0.03,
            'profit_factor': 2.0,
            'trade_count': 50,
        }
        score = scorer._score_backtest(metrics)

        # Should be a reasonably high score
        assert score > 50

    @patch('manager.pair_scorer.OrderManager')
    @patch('manager.pair_scorer.Database')
    def test_score_backtest_insufficient_data(self, mock_db, mock_om, mock_config):
        """Test backtest scoring with insufficient data returns neutral."""
        from manager.pair_scorer import PairScorer

        mock_om_instance = MagicMock()
        mock_om.return_value = mock_om_instance
        mock_db.return_value = MagicMock()

        scorer = PairScorer(mock_config)

        metrics = {'trade_count': 5}  # Too few trades
        score = scorer._score_backtest(metrics)

        assert score == 50.0  # Neutral

    @patch('manager.pair_scorer.OrderManager')
    @patch('manager.pair_scorer.Database')
    def test_fetch_backtest_metrics(self, mock_db, mock_om, mock_config):
        """Test fetching backtest metrics from database."""
        from manager.pair_scorer import PairScorer

        mock_om_instance = MagicMock()
        mock_om.return_value = mock_om_instance

        expected_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': 0.03,
            'profit_factor': 2.0,
            'avg_trade_return': 0.005,
            'trade_count': 50,
        }
        mock_db_instance = MagicMock()
        mock_db_instance.get_backtest_metrics.return_value = expected_metrics
        mock_db.return_value = mock_db_instance

        scorer = PairScorer(mock_config)
        metrics = scorer._fetch_backtest_metrics('SOL/USDT')

        assert metrics == expected_metrics
        mock_db_instance.get_backtest_metrics.assert_called_once_with('SOL/USDT', 30)
