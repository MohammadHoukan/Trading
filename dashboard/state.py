import json


DEFAULT_STALE_AFTER_SECONDS = 30


def broadcast_stop(redis_client, command_channel):
    """Broadcast a STOP-all command and return (ok, message)."""
    try:
        receivers = redis_client.publish(
            command_channel,
            json.dumps({'command': 'STOP', 'target': 'all'}),
        )
    except Exception as exc:
        return False, f"Failed to broadcast STOP: {exc}"

    if isinstance(receivers, int) and receivers > 0:
        return True, f"STOP delivered to {receivers} subscriber(s)."
    return False, "STOP not delivered: no active command subscribers."


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_worker_rows(workers_raw, now_ts, stale_after_seconds=DEFAULT_STALE_AFTER_SECONDS):
    """
    Parse worker hash payloads into active/stale row sets.

    Returns: (active_rows, stale_rows, malformed_count)
    """
    if not workers_raw:
        return [], [], 0

    active_rows = []
    stale_rows = []
    malformed_count = 0

    for worker_id, worker_json in workers_raw.items():
        try:
            payload = json.loads(worker_json)
        except (TypeError, json.JSONDecodeError):
            malformed_count += 1
            continue

        if not isinstance(payload, dict):
            malformed_count += 1
            continue

        row = dict(payload)
        # Ignore payload 'worker_id' for identity to prevent spoofing.
        # Use the Redis hash key as the source of truth.
        if 'worker_id' in row:
            del row['worker_id']
        row['worker_id'] = worker_id

        last_updated = _to_float(row.get('last_updated'))
        age_seconds = None if last_updated is None else max(0.0, now_ts - last_updated)
        row['age_seconds'] = age_seconds

        is_stale = last_updated is None or age_seconds > stale_after_seconds
        row['is_stale'] = is_stale

        if is_stale:
            stale_rows.append(row)
        else:
            active_rows.append(row)

    return active_rows, stale_rows, malformed_count
