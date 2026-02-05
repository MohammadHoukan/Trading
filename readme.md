# 🌐 Spot-Grid-Swarm

**A High-Performance, Distributed Multi-Agent Trading System for Spot Market Grid Strategies.**

---

## 📖 Overview

**Spot-Grid-Swarm** is an institutional-grade trading architecture designed to orchestrate a cluster of independent grid trading bots ("Workers") from a centralized "Manager". It bridges the gap between simple grid bots and professional multi-agent systems.

### 🛡️ Core Constraints
1. **Spot Market Only:** CCXT wrapper prevents any interaction with Futures/Derivatives.
2. **Zero Leverage:** 1:1 capital basis (no margin borrowing).
3. **Long-Only:** Accumulates and sells the underlying asset without short-selling risk.
4. **Execution Drag Awareness:** Backtester simulates slippage, spreads, and partial fills to ensure "backtest truthfulness".

---

## 🏗 Architecture

The system utilizes a **Hub-and-Spoke** architecture with **Redis Streams** for reliable inter-process communication.

### 👑 The Manager (Orchestrator)
The "Brain" of the swarm. It doesn't trade directly but manages the health and logic of the workers.
- **Per-Symbol Regime Detection:** Analyzes market conditions for each active pair independently.
- **Reliable STOP Broadcast:** Throttled retry mechanism continuously asserts `STOP` commands during risk breaches to ensure all workers halt.
- **Composite Scoring:** Uses a weighted average of ADX (Trend), ATR (Volatility), MA Distance (Mean Reversion), and historical Fill Rate (Execution) to decide if a strategy should `RUN`, `HOLD`, or `PAUSE`.
- **Global Risk Engine:** Enforces concurrency limits and global capital allocation thresholds.

### 🐝 The Workers (Swarm)
Independent processes spawned per trading pair.
- **Graceful Shutdown:** Handles `SIGINT` (Ctrl+C) and `SIGTERM` signals to cleanly unregister from the manager, preventing "zombie" risk quotas.
- **Dynamic Key Pool:** Prevents nonce collisions by claiming API keys from a Redis-locked pool.
- **Rolling (Infinity) Grids:** Grid levels shift dynamically with price to prevent "trading out" of the range.
- **Stop-Loss Protection:** Hard-coded price floor that cancels all orders and unregisters the bot.
- **Watchdog:** Monitors connection health and stalls trading on stale price data.

### 📡 The Bus (Redis)
- **Messaging:** Uses Pub/Sub for low-latency broadcasts and Streams for reliable command delivery.
- **State:** Stores worker snapshots and API key locks.

---

## 🎛 Regime Detection Logic

The `RegimeFilter` computes a **Composite Score (0-100)**:
- **Score ≥ 60 (RANGING):** Optimal for grid. Sends `RESUME` command.
- **Score 40-60 (UNCERTAIN):** Sends `HOLD`. Keeps current execution state.
- **Score < 40 (TRENDING):** High risk for grid. Sends `PAUSE` command to stall order placement.

---

## 🔬 Tooling & Optimization

The system includes a suite of tools for strategy optimization and pair selection:

### 1. Pair Scorer (`manager/pair_scorer.py`)
Systematically evaluates pairs based on liquidity, bandwidth, fees, and effective historical fill rates.
```bash
python3 manager/pair_scorer.py SOL/USDT ETH/USDT
```

### 2. Parameter Optimizer (`backtest/optimizer.py`)
Runs parameter sweeps to find the "Sweet Spot" for grid spacing and capital allocation over a given period.
```bash
# Optimize settings for a specific pair
python3 -m backtest.optimizer --pair SOL/USDT --days 30 --save
```

### 3. Regime Validator (`backtest/regime_validator.py`)
Validates whether the regime filter strategy actually improves profitability by comparing "Always Trade" vs "Regime Filtered" simulations.
```bash
python3 -m backtest.regime_validator --pair SOL/USDT --days 60 --verbose
```

---

## 📊 Backtesting & Verification

We support three levels of strategy validation:

### 1. Realistic Simulation
The `ExecutionModel` simulates real-world frictions:
- **Slippage:** Exponential distribution of fill prices moving against the bot.
- **Spread:** Bid-ask simulation.
- **Partial Fills:** beta-distribution of order fulfillment (30-100%).

```bash
# Run realistic backtest with Execution Drag enabled
python3 -m backtest.runner --pair SOL/USDT --realistic
```

### 2. Portfolio Backtesting
Aggregate results across multiple concurrent strategies to see total correlation and capital drag.
```bash
python3 -m backtest.portfolio_runner --days 30 --capital 1000 --realistic
```

---

## 📂 Project Structure

```bash
├── manager/
│   ├── orchestrator.py    # Brain; manages heartbeats & targeted commands
│   ├── regime_filter.py   # Composite signal analysis (ADX, ATR, MA, Fill Rate)
│   ├── risk_engine.py     # Capital & concurrency limits
│   └── pair_scorer.py     # Liquidity & Volatility scoring for pair selection
├── workers/
│   ├── grid_bot.py        # execution engine; Key Pool & Rolling Grid support
│   └── order_manager.py   # Strict Spot-only CCXT abstraction
├── shared/
│   ├── messaging.py       # Redis Stream + Pub/Sub wrappers
│   ├── database.py        # SQLite logger (live vs backtest isolation)
│   └── config.py          # .env + ${VAR} resolution
├── backtest/
│   ├── simulator.py       # Core logic with rolling grid support
│   ├── optimizer.py       # Parameter sweep tool
│   ├── regime_validator.py# Logic validation tool
│   └── execution_model.py # Slippage/Spread simulation
├── config/
│   ├── settings.yaml      # API Keys (Pool format) & System Limits
│   └── strategies.json    # Grid parameters per pair
└── dashboard/             # Streamlit-based Control Room
```

---

## 🚀 Getting Started

### 1. Installation
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Setup your `.env` and `config/settings.yaml`. Use the **API Key Pool** for distributed scaling:
```yaml
exchange:
  pool:
    - api_key: "KEY_1"
      secret: "SEC_1"
    - api_key: "KEY_2"
      secret: "SEC_2"
```

### 3. Operations
1. **Start Redis:** `redis-server`
2. **Launch Manager:** `python3 manager/orchestrator.py`
3. **Spawn Workers:** `python3 workers/grid_bot.py --pair SOL/USDT --grids 15`
4. **View Dashboard:** `streamlit run dashboard/app.py`

---

## 🤝 Contributing & License
Alpha software. No warranty provided. Thorough backtesting required.
For major changes, please open an issue or submit a PR with verification logs.
