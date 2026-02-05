import os
import re
import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r'\$\{(\w+)(?::-([^}]*))?\}')


def _resolve_env(raw_text):
    def replace(match):
        var_name = match.group(1)
        default_val = match.group(2)
        env_val = os.getenv(var_name)
        if env_val is not None:
            return env_val
        if default_val is not None:
            return default_val
        return match.group(0)

    return _ENV_PATTERN.sub(replace, raw_text)


def _coerce_int(value, path):
    if value is None:
        raise ValueError(f"Missing required int for {path}")
    if isinstance(value, bool):
        raise ValueError(f"Invalid bool for {path}: {value}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("${") and candidate.endswith("}"):
            raise ValueError(f"Unresolved env placeholder for {path}: {candidate}")
        try:
            return int(candidate)
        except ValueError as exc:
            raise ValueError(f"Invalid int for {path}: {value}") from exc
    raise ValueError(f"Invalid int for {path}: {value}")


def _coerce_float(value, path):
    if value is None:
        raise ValueError(f"Missing required float for {path}")
    if isinstance(value, bool):
        raise ValueError(f"Invalid bool for {path}: {value}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("${") and candidate.endswith("}"):
            raise ValueError(f"Unresolved env placeholder for {path}: {candidate}")
        try:
            return float(candidate)
        except ValueError as exc:
            raise ValueError(f"Invalid float for {path}: {value}") from exc
    raise ValueError(f"Invalid float for {path}: {value}")


def normalize_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a dict")

    redis_cfg = config.get('redis', {})
    if isinstance(redis_cfg, dict):
        if 'port' in redis_cfg:
            redis_cfg['port'] = _coerce_int(redis_cfg['port'], 'redis.port')
        if 'db' in redis_cfg:
            redis_cfg['db'] = _coerce_int(redis_cfg['db'], 'redis.db')

    swarm_cfg = config.get('swarm', {})
    if isinstance(swarm_cfg, dict):
        if 'max_concurrency' in swarm_cfg:
            swarm_cfg['max_concurrency'] = _coerce_int(
                swarm_cfg['max_concurrency'], 'swarm.max_concurrency'
            )
        if 'risk_per_bot' in swarm_cfg:
            swarm_cfg['risk_per_bot'] = _coerce_float(
                swarm_cfg['risk_per_bot'], 'swarm.risk_per_bot'
            )
        if 'loop_interval' in swarm_cfg:
            swarm_cfg['loop_interval'] = _coerce_int(
                swarm_cfg['loop_interval'], 'swarm.loop_interval'
            )

    dashboard_cfg = config.get('dashboard', {})
    if isinstance(dashboard_cfg, dict):
        if 'worker_stale_after_seconds' in dashboard_cfg:
            dashboard_cfg['worker_stale_after_seconds'] = _coerce_int(
                dashboard_cfg['worker_stale_after_seconds'],
                'dashboard.worker_stale_after_seconds',
            )

    return config


def get_redis_params(config):
    redis_cfg = config.get('redis', {})
    return {
        'host': redis_cfg.get('host', 'localhost'),
        'port': redis_cfg.get('port', 6379),
        'db': redis_cfg.get('db', 0),
    }


def load_config(config_path='config/settings.yaml'):
    load_dotenv()

    resolved_path = config_path
    if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidate = os.path.join(repo_root, config_path)
        if os.path.exists(candidate):
            resolved_path = candidate

    with open(resolved_path) as f:
        raw_config = f.read()

    resolved_config = _resolve_env(raw_config)
    parsed = yaml.safe_load(resolved_config)
    
    # Extra check: Ensure pool items are fully resolved if YAML parsing missed nested structures
    # (Though _resolve_env should handle raw text, this acts as a sanitizer for any keys added later)
    return normalize_config(parsed)
