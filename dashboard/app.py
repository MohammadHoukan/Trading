import streamlit as st
import pandas as pd
import redis
import os
import sys
import time

# Hack to add root path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.config import load_config, get_redis_params
from dashboard.state import (
    broadcast_stop,
    parse_worker_rows,
    DEFAULT_STALE_AFTER_SECONDS,
)

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
dashboard_cfg = config.get('dashboard', {})
stale_after_seconds = dashboard_cfg.get(
    'worker_stale_after_seconds',
    DEFAULT_STALE_AFTER_SECONDS,
)
if not isinstance(stale_after_seconds, int) or stale_after_seconds <= 0:
    stale_after_seconds = DEFAULT_STALE_AFTER_SECONDS

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
        if not r:
            st.warning("Redis not connected. Cannot broadcast STOP.")
        else:
            ok, message = broadcast_stop(r, config['redis']['channels']['command'])
            if ok:
                st.error(f"STOP SIGNAL BROADCASTED! {message}")
            else:
                st.warning(message)

with col2:
    st.header("Active Workers")
    
    if r:
        try:
            # Fetch all worker data from Redis Hash
            workers_raw = r.hgetall('workers:data')
            active_workers, stale_workers, malformed_count = parse_worker_rows(
                workers_raw,
                now_ts=time.time(),
                stale_after_seconds=stale_after_seconds,
            )

            if active_workers:
                df = pd.DataFrame(active_workers)

                cols = [
                    'worker_id',
                    'symbol',
                    'status',
                    'inventory',
                    'avg_cost',
                    'realized_profit',
                    'price',
                    'exposure',
                    'last_updated',
                    'age_seconds',
                ]
                existing_cols = [c for c in cols if c in df.columns]
                st.dataframe(df[existing_cols] if existing_cols else df, use_container_width=True)
            else:
                st.info("No active workers found in Redis.")

            if stale_workers:
                st.warning(
                    f"{len(stale_workers)} stale worker(s) hidden "
                    f"(older than {stale_after_seconds}s)."
                )
                with st.expander("Show stale workers"):
                    stale_df = pd.DataFrame(stale_workers)
                    stale_cols = [
                        'worker_id',
                        'symbol',
                        'status',
                        'last_updated',
                        'age_seconds',
                    ]
                    stale_existing_cols = [c for c in stale_cols if c in stale_df.columns]
                    st.dataframe(
                        stale_df[stale_existing_cols] if stale_existing_cols else stale_df,
                        use_container_width=True,
                    )

            if malformed_count > 0:
                st.warning(f"Skipped {malformed_count} malformed worker record(s).")
                 
        except Exception as e:
            st.error(f"Error fetching worker data: {e}")
    else:
        st.warning("Redis not connected. Cannot fetch worker data.")
    
    # Auto-refresh mechanism (simple button for now, or use st.empty loop in a real app)
    if st.button("Refresh Data"):
        st.rerun()

st.subheader("System Config")
st.json(_redact_config(config))
