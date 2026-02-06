"""
ML module for smart trading features.

This module provides:
- Regime classification using Random Forest
- Pair profitability prediction using Gradient Boosting
- Feature engineering for market analysis
- Model persistence and training utilities
"""

from ml.models.regime_classifier import RegimeClassifier
from ml.models.pair_predictor import PairPredictor

__all__ = ['RegimeClassifier', 'PairPredictor']
