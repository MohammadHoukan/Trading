"""Smoke test for config parsing."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import load_config


def test_config_loads():
    """Config file loads without error."""
    config = load_config()
    
    assert config is not None


def test_config_has_required_keys():
    """Config has all required top-level keys."""
    config = load_config()
    
    assert 'exchange' in config
    assert 'redis' in config
    assert 'swarm' in config


def test_redis_config_has_connection_params():
    """Redis config has host, port, db for connection."""
    config = load_config()
    
    redis_cfg = config['redis']
    assert 'host' in redis_cfg
    assert 'port' in redis_cfg
    assert 'db' in redis_cfg


def test_redis_config_has_channels():
    """Redis config has channels for pub/sub."""
    config = load_config()
    
    assert 'channels' in config['redis']
    channels = config['redis']['channels']
    assert 'command' in channels
    assert 'status' in channels
