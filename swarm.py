#!/usr/bin/env python3
"""
Swarm Management CLI - Easy control for paper trading.

Usage:
    python swarm.py start          # Start orchestrator + all enabled workers
    python swarm.py stop           # Stop all workers gracefully
    python swarm.py status         # Show status of all components
    python swarm.py dashboard      # Launch Streamlit dashboard
    python swarm.py logs           # Tail combined logs
    python swarm.py train          # Retrain ML models with latest data
"""

import subprocess
import sys
import os
import time
import json
import signal
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from shared.config import load_config
from shared.messaging import RedisBus

PID_DIR = "/tmp/swarm_pids"
LOG_DIR = "logs"


def ensure_dirs():
    os.makedirs(PID_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def save_pid(name, pid):
    with open(f"{PID_DIR}/{name}.pid", "w") as f:
        f.write(str(pid))


def get_pid(name):
    try:
        with open(f"{PID_DIR}/{name}.pid", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None


def is_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_strategies():
    with open("config/strategies.json", "r") as f:
        return json.load(f)


def check_redis():
    """Check if Redis is running."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True
    except Exception:
        return False


def start_redis():
    """Start Redis if not running."""
    if check_redis():
        print("✓ Redis already running")
        return True

    # Try different methods to start Redis
    methods = [
        # Method 1: Direct redis-server
        (["redis-server", "--daemonize", "yes"], "redis-server"),
        # Method 2: systemctl
        (["systemctl", "start", "redis-server"], "systemctl"),
        # Method 3: service
        (["service", "redis-server", "start"], "service"),
    ]

    for cmd, method in methods:
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
            time.sleep(1)
            if check_redis():
                print(f"✓ Redis started via {method}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # If all methods fail, give instructions
    print("✗ Could not start Redis automatically.")
    print("\nTo start Redis, run ONE of these in another terminal:")
    print("  Option 1: sudo apt install redis-server && sudo systemctl start redis-server")
    print("  Option 2: docker run -d --name redis -p 6379:6379 redis:alpine")
    print("  Option 3: redis-server (if installed)")
    return False


def cmd_start(args):
    """Start the swarm (orchestrator + workers)."""
    ensure_dirs()

    # Check/start Redis
    if not start_redis():
        print("Cannot start without Redis. Install with: sudo apt install redis-server")
        return 1

    # Load config and strategies
    config = load_config()
    
    # Clean slate: Remove stale commands from previous run
    try:
        from shared.config import get_redis_params
        import redis
        r = redis.Redis(**get_redis_params(config))
        r.delete('swarm:commands')
        print("✓ Cleared stale commands from Redis")
    except Exception as e:
        print(f"Warning: Could not clear Redis stream: {e}")

    strategies = load_strategies()
    enabled_pairs = [s for s, cfg in strategies.items() if cfg.get('enabled', True)]

    print(f"\nEnabled pairs: {', '.join(enabled_pairs)}")
    print(f"Mode: {config['exchange'].get('mode', 'testnet')}")

    # Start orchestrator
    orch_pid = get_pid("orchestrator")
    if is_running(orch_pid):
        print("✓ Orchestrator already running")
    else:
        print("Starting orchestrator...")
        log_file = open(f"{LOG_DIR}/orchestrator.log", "a")
        proc = subprocess.Popen(
            [sys.executable, "manager/orchestrator.py"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        save_pid("orchestrator", proc.pid)
        print(f"✓ Orchestrator started (PID: {proc.pid})")

    time.sleep(2)  # Let orchestrator initialize

    # Start workers
    for pair in enabled_pairs:
        worker_name = f"worker_{pair.replace('/', '_')}"
        worker_pid = get_pid(worker_name)

        if is_running(worker_pid):
            print(f"✓ {pair} worker already running")
            continue

        grids = strategies[pair].get('grid_levels', 20)
        print(f"Starting {pair} worker ({grids} grids)...")

        log_file = open(f"{LOG_DIR}/{worker_name}.log", "a")
        proc = subprocess.Popen(
            [sys.executable, "workers/grid_bot.py", "--pair", pair, "--grids", str(grids)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        save_pid(worker_name, proc.pid)
        print(f"✓ {pair} worker started (PID: {proc.pid})")
        time.sleep(1)  # Stagger starts to avoid API key race

    print(f"\n✓ Swarm started! Monitor with: python swarm.py status")
    print(f"  Dashboard: python swarm.py dashboard")
    print(f"  Logs: python swarm.py logs")
    return 0


def cmd_stop(args):
    """Stop all swarm components gracefully."""
    ensure_dirs()

    # Send STOP command via Redis
    if check_redis():
        try:
            from shared.messaging import RedisBus
            from shared.config import get_redis_params, load_config
            config = load_config()
            bus = RedisBus(**get_redis_params(config))
            bus.publish(config['redis']['channels']['command'], {'command': 'STOP', 'target': 'all'})
            print("Sent STOP command to all workers")
        except Exception as e:
            print(f"Warning: Could not send STOP command: {e}")

    time.sleep(2)

    # Kill processes
    strategies = load_strategies()
    enabled_pairs = [s for s, cfg in strategies.items() if cfg.get('enabled', True)]

    # Stop workers first
    for pair in enabled_pairs:
        worker_name = f"worker_{pair.replace('/', '_')}"
        pid = get_pid(worker_name)
        if is_running(pid):
            print(f"Stopping {pair} worker (PID: {pid})...")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.remove(f"{PID_DIR}/{worker_name}.pid")
            except FileNotFoundError:
                pass

    time.sleep(2)

    # Stop orchestrator
    orch_pid = get_pid("orchestrator")
    if is_running(orch_pid):
        print(f"Stopping orchestrator (PID: {orch_pid})...")
        try:
            os.kill(orch_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            os.remove(f"{PID_DIR}/orchestrator.pid")
        except FileNotFoundError:
            pass

    print("✓ Swarm stopped")
    return 0


def cmd_status(args):
    """Show status of all components."""
    ensure_dirs()

    print("\n" + "=" * 80)
    print(f"{'SWARM STATUS':^80}")
    print("=" * 80)

    # Redis
    redis_alive = check_redis()
    redis_status = "✓ Running" if redis_alive else "✗ Not running"
    print(f"Redis:        {redis_status}")

    # Orchestrator
    orch_pid = get_pid("orchestrator")
    if is_running(orch_pid):
        orch_status = f"✓ Running (PID: {orch_pid})" 
    elif orch_pid:
        orch_status = f"☠ Dead (PID file {orch_pid} exists but process gone)"
    else:
        orch_status = "✗ Not running"
    print(f"Orchestrator: {orch_status}")

    # Load Strategies & Redis Data
    strategies = load_strategies()
    workers_redis = {}
    if redis_alive:
        try:
            from shared.messaging import RedisBus
            from shared.config import get_redis_params, load_config
            import time
            
            config = load_config()
            bus = RedisBus(**get_redis_params(config))
            raw_data = bus.hgetall('workers:data')
            if raw_data:
                for k, v in raw_data.items():
                    try:
                        workers_redis[k] = json.loads(v)
                    except:
                        pass
        except Exception:
            pass

    print("\nWorkers:")
    print("-" * 80)
    print(f"{'Pair':<12} {'PID':<10} {'State':<15} {'Redis Heartbeat':<20} {'Info'}")
    print("-" * 80)

    # 1. Check Configured Workers
    for pair, cfg in strategies.items():
        worker_name = f"worker_{pair.replace('/', '_')}"
        pid = get_pid(worker_name)
        enabled = cfg.get('enabled', True)
        
        # Match with Redis data (find worker with this symbol)
        redis_info = None
        worker_id = None
        for wid, wdata in workers_redis.items():
            if wdata.get('symbol') == pair:
                redis_info = wdata
                worker_id = wid
                break

        status_icon = "✗"
        state_text = "Stopped"
        pid_text = "-"
        hb_text = "-"
        info_text = ""

        if not enabled:
            status_icon = "⊘"
            state_text = "Disabled"
        else:
            is_alive = is_running(pid)
            
            # HB Check
            last_hb_age = None
            if redis_info:
                last_updated = redis_info.get('last_updated', 0)
                last_hb_age = time.time() - last_updated
                hb_text = f"{last_hb_age:.1f}s ago"
            
            if is_alive:
                pid_text = str(pid)
                status_icon = "✓"
                state_text = "Running"
                
                # Zombie Check: Running locally but no/stale heartbeat
                if not redis_info:
                     state_text = "ZOMBIE?"
                     info_text = "Running but not in Redis"
                elif last_hb_age and last_hb_age > 30:
                     state_text = "HANGING?"
                     info_text = f"Stale heartbeat (>30s)"
            else:
                if pid:
                    status_icon = "☠"
                    state_text = "Dead"
                    pid_text = f"{pid}(?)"
                    info_text = "PID file exists, process gone"
                elif redis_info and redis_info.get('status') == 'RUNNING':
                     status_icon = "👻"
                     state_text = "GHOST"
                     info_text = "In Redis but no local PID (Remote?)"

        print(f"{status_icon} {pair:<10} {pid_text:<10} {state_text:<15} {hb_text:<20} {info_text}")

    print("=" * 80)
    return 0


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    print("Launching dashboard...")
    print("Press Ctrl+C to stop\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    return 0


def cmd_logs(args):
    """Tail combined logs."""
    ensure_dirs()
    log_files = [f"{LOG_DIR}/{f}" for f in os.listdir(LOG_DIR) if f.endswith('.log')]

    if not log_files:
        print("No log files found. Start the swarm first.")
        return 1

    print(f"Tailing {len(log_files)} log files. Press Ctrl+C to stop.\n")
    try:
        subprocess.run(["tail", "-f"] + log_files)
    except KeyboardInterrupt:
        print("\nStopped")
    return 0


def cmd_train(args):
    """Retrain ML models with latest data."""
    print("Retraining ML models...\n")

    # Train regime classifier
    print("=" * 40)
    print("Training Regime Classifier")
    print("=" * 40)
    result = subprocess.run([
        sys.executable, "-m", "ml.training.regime_trainer",
        "--symbol", "SOL/USDT", "--days", "30", "--save"
    ])

    print("\n" + "=" * 40)
    print("Training Pair Predictor")
    print("=" * 40)
    result = subprocess.run([
        sys.executable, "-m", "ml.training.pair_trainer",
        "--days", "30", "--save"
    ])

    print("\n✓ Training complete!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Swarm Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("start", help="Start orchestrator and workers")
    subparsers.add_parser("stop", help="Stop all components")
    subparsers.add_parser("status", help="Show status")
    subparsers.add_parser("dashboard", help="Launch dashboard")
    subparsers.add_parser("logs", help="Tail logs")
    subparsers.add_parser("train", help="Retrain ML models")

    args = parser.parse_args()

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "dashboard": cmd_dashboard,
        "logs": cmd_logs,
        "train": cmd_train,
    }

    if args.command is None:
        parser.print_help()
        return 0

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
