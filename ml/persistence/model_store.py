"""
Model persistence utilities for saving and loading trained models.

Uses joblib for efficient serialization of scikit-learn models.
"""

import os
import logging
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available, model persistence disabled")


def save_model(
    model: Any,
    path: str,
    metadata: Optional[dict] = None,
    compress: int = 3,
) -> bool:
    """
    Save a trained model to disk.

    Args:
        model: Trained model object (sklearn estimator)
        path: File path to save to (should end in .joblib)
        metadata: Optional metadata to save with the model
        compress: Compression level (0-9, default 3)

    Returns:
        True if successful, False otherwise
    """
    if not JOBLIB_AVAILABLE:
        logger.error("Cannot save model: joblib not installed")
        return False

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # Package model with metadata
        package = {
            'model': model,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat(),
        }

        joblib.dump(package, path, compress=compress)
        logger.info(f"Model saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        return False


def load_model(path: str) -> Optional[Any]:
    """
    Load a trained model from disk.

    Args:
        path: File path to load from

    Returns:
        Model object or None if failed
    """
    if not JOBLIB_AVAILABLE:
        logger.error("Cannot load model: joblib not installed")
        return None

    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}")
        return None

    try:
        package = joblib.load(path)

        if isinstance(package, dict) and 'model' in package:
            logger.info(
                f"Loaded model from {path} (saved: {package.get('saved_at', 'unknown')})"
            )
            return package['model']
        else:
            # Legacy format - just the model
            logger.info(f"Loaded model from {path} (legacy format)")
            return package

    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        return None


def load_model_with_metadata(path: str) -> tuple:
    """
    Load a model along with its metadata.

    Args:
        path: File path to load from

    Returns:
        Tuple of (model, metadata) or (None, None) if failed
    """
    if not JOBLIB_AVAILABLE:
        logger.error("Cannot load model: joblib not installed")
        return None, None

    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}")
        return None, None

    try:
        package = joblib.load(path)

        if isinstance(package, dict) and 'model' in package:
            return package['model'], package.get('metadata', {})
        else:
            return package, {}

    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        return None, None


def model_exists(path: str) -> bool:
    """Check if a model file exists."""
    return os.path.exists(path)


def get_model_info(path: str) -> Optional[dict]:
    """
    Get metadata about a saved model without loading it fully.

    Args:
        path: File path

    Returns:
        Dict with model info or None
    """
    if not os.path.exists(path):
        return None

    try:
        # Try to load just the metadata
        # Note: This still loads the full file, but we only return metadata
        package = joblib.load(path)

        if isinstance(package, dict):
            return {
                'saved_at': package.get('saved_at'),
                'metadata': package.get('metadata', {}),
                'has_model': 'model' in package,
            }
        else:
            return {
                'saved_at': None,
                'metadata': {},
                'has_model': True,
            }

    except Exception as e:
        logger.warning(f"Failed to get model info from {path}: {e}")
        return None
