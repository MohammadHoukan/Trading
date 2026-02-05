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
    return yaml.safe_load(resolved_config)
