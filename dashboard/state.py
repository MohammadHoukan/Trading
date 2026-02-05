import json


DEFAULT_STALE_AFTER_SECONDS = 30
COMMAND_STREAM = 'swarm:commands'


def broadcast_stop(redis_client, command_channel, command_stream=COMMAND_STREAM):
    """Broadcast a STOP-all command and return (ok, message)."""
    payload = {'command': 'STOP', 'target': 'all'}
    stream_ok = False
    pubsub_ok = False
    pubsub_receivers = 0
    errors = []

    try:
        stream_id = redis_client.xadd(command_stream, payload, maxlen=1000)
        stream_ok = stream_id is not None
    except Exception as exc:
        errors.append(f"stream enqueue failed: {exc}")

    try:
        pubsub_receivers = redis_client.publish(command_channel, json.dumps(payload))
        pubsub_ok = isinstance(pubsub_receivers, int) and pubsub_receivers > 0
    except Exception as exc:
        errors.append(f"pubsub broadcast failed: {exc}")

    if stream_ok and pubsub_ok:
        return True, f"STOP enqueued and delivered to {pubsub_receivers} subscriber(s)."
    if stream_ok:
        return True, "STOP enqueued to reliable stream; workers will consume on reconnect."
    if pubsub_ok:
        return True, f"STOP delivered to {pubsub_receivers} subscriber(s)."
    if errors:
        return False, f"Failed to broadcast STOP: {'; '.join(errors)}"
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
