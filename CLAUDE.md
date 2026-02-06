# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spot-Grid-Swarm is a distributed multi-agent trading system for cryptocurrency spot market grid strategies. It uses a Hub-and-Spoke architecture with Redis for inter-process communication.

**Core Constraints:** Spot-only (no futures/derivatives), zero leverage, long-only positions.

## Commands

### Running the System
```bash
# Prerequisites: Redis must be running
redis-server

# Start the orchestrator (manager)
python manager/orchestrator.py

# Start a worker for a specific trading pair
python workers/grid_bot.py --pair SOL/USDT --grids 20

# Launch the dashboard
streamlit run dashboard/app.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_config.py

# Run a specific test
pytest tests/test_config.py::test_config_loads
```

### Backtesting & Tools
```bash
# Run backtest with realistic execution model
python -m backtest.runner --pair SOL/USDT --realistic

# Portfolio backtest
python -m backtest.portfolio_runner --days 30 --capital 1000 --realistic

# Parameter optimization
python -m backtest.optimizer --pair SOL/USDT --days 30 --save

# Regime filter validation
python -m backtest.regime_validator --pair SOL/USDT --days 60 --verbose

# Pair scoring for selection
python manager/pair_scorer.py SOL/USDT ETH/USDT
```

## Architecture

### Component Communication Flow
```
Orchestrator (Manager)
    ├── Risk Engine: drawdown protection, capital allocation
    ├── Regime Filter: market analysis, exposure scaling
    └── Market Data Publisher: broadcasts prices every 2s
            │
    Redis (Pub/Sub + Streams + Hash)
            │
    GridBot Workers (one per trading pair)
        ├── Order placement/monitoring
        ├── Grid calculation & rebalancing
        └── Status reporting every 2s
            │
    Binance Exchange (via CCXT)
```

### Redis Channels
- `swarm:cmd` - Commands from orchestrator to workers (STOP, PAUSE, RESUME, UPDATE_PARAMS, UPDATE_SCALE)
- `swarm:status` - Worker status updates to orchestrator
- `market_data:{symbol}` - Price broadcasts from orchestrator
- `workers:data` - Hash storing persistent worker state snapshots

### Key Patterns

**API Key Pool:** Workers acquire unique API key locks from Redis to prevent nonce collisions. Lock renewal thread maintains ownership.

**Rolling Grids:** Grid boundaries shift dynamically when price breaks limits, enabling continuous trading in trending markets.

**Dual-Channel Messaging:** Pub/Sub for real-time delivery; Redis Streams for guaranteed delivery on reconnect.

**Exposure Scaling:** UPDATE_SCALE adjusts position sizing in-place without rebuilding the grid. Used for regime-based exposure adjustment.

### Risk Management
- Per-worker capital limit (`risk_per_bot`)
- Global capital allocation threshold
- Drawdown-based actions: 10% warning → 15% reduce exposure → 20% halt all
- Per-worker stop-loss price floors

## Configuration

- `config/settings.yaml` - API keys (pool format), Redis config, swarm limits, regime settings
- `config/strategies.json` - Grid parameters per trading pair (levels, limits, amount, stop-loss)
- Environment variables supported via `${VAR_NAME:-default}` syntax in YAML

## Key Files

| File | Purpose |
|------|---------|
| `manager/orchestrator.py` | Central controller, risk checks, regime analysis |
| `manager/risk_engine.py` | Drawdown calculation, capital allocation |
| `manager/regime_filter.py` | ADX/ATR/MA-based market regime detection |
| `workers/grid_bot.py` | Individual trading worker, grid logic |
| `workers/order_manager.py` | CCXT wrapper, spot-only enforcement |
| `shared/messaging.py` | Redis Pub/Sub + Streams abstraction |
| `shared/rate_limiter.py` | Redis-backed token bucket (8 req/sec default) |
| `backtest/simulator.py` | Grid trading simulation with rolling grid support |
