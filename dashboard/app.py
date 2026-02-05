import streamlit as st
import pandas as pd
import json
import redis
import os
import sys

# Hack to add root path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config import load_config, get_redis_params

def _redact_config(value):
    if isinstance(value, dict):
        redacted = {}
        for key, val in value.items():
            key_lower = key.lower() if isinstance(key, str) else key
            if isinstance(key_lower, str) and (
                'secret' in key_lower or 'token' in key_lower or 'key' in key_lower
            ):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_config(val)
        return redacted
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value

config = load_config()

st.set_page_config(page_title="Spot-Grid-Swarm Control Room", layout="wide")
st.title("🐝 Spot-Grid-Swarm Control Room")

# Redis Connection
r = None
try:
    redis_params = get_redis_params(config)
    r = redis.Redis(
        host=redis_params['host'],
        port=redis_params['port'],
        db=redis_params['db'],
        decode_responses=True,
    )
    r.ping()  # Verify connection works
    st.sidebar.success("Connected to Redis")
except Exception as e:
    r = None
    st.sidebar.error(f"Redis Error: {e}")

# Main Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Global Controls")
    if st.button("🚨 EMERGENCY KILL SWITCH", disabled=(r is None)):
        if r:
            r.publish(config['redis']['channels']['command'], json.dumps({'command': 'STOP', 'target': 'all'}))
            st.error("STOP SIGNAL BROADCASTED!")

with col2:
    st.header("Active Workers")
    # This would actually need to read from SQLite for persistence or listen to a stream
    # For now, just a placeholder explaining architecture
    st.info("Worker status is logged to SQLite and broadcast via Redis Pub/Sub.")
    st.markdown("""
    **To view live worker data, ensuring the Orchestrator is saving heartbeats to DB is required.**
    """)

st.subheader("System Config")
st.json(_redact_config(config))
