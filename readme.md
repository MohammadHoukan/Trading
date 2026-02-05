# 🌐 Spot-Grid-Swarm

**A Distributed, Multi-Agent Trading System for Spot Market Grid Strategies.**

## 📖 Overview

**Spot-Grid-Swarm** is a Python-based trading architecture designed to orchestrate a cluster of independent grid trading bots.

This system is engineered with strict **Risk & Asset Constraints**:

1. **Spot Market Only:** No interaction with Futures, Options, or Derivatives.
2. **Zero Leverage:** Operates strictly on a 1:1 capital basis (no margin borrowing).
3. **Long-Only:** No short selling. The system focuses on accumulating and selling the underlying asset.

It utilizes a **Hub-and-Spoke** architecture to manage concurrency, allowing multiple assets to be traded simultaneously while maintaining a global risk state. Grid execution is basic (initial placement + rebalancing), while risk controls remain stubs.

## 🏗 Architecture

The system mimics a microservices pattern using **Redis Pub/Sub** for inter-process communication:

* **👑 The Manager (Orchestrator):**
* Subscribes to worker status updates and can broadcast commands (e.g., STOP).
* Regime detection and risk checks are placeholders.


* **🐝 The Workers (Swarm):**
* Independent processes spawned per trading pair (e.g., `SOL/USDT`, `ETH/USDT`).
* **Dynamic Key Pool:** Workers "claim" a unique API Key from Redis at startup to prevent Nonce collisions.
* Place initial grid orders and rebalance on fills (basic implementation).


* **📡 The Bus (Redis):**
* Facilitates low-latency messaging between Manager and Workers.
* Used for command dispatch (Start/Stop) and status reporting.



## ⚙️ Core System Rules

The following constraints are hard-coded into the `order_manager.py` logic:

* **Rule 1: No Derivatives.** The API connector is forced to `defaultType: 'spot'`. Any request to a non-spot endpoint is rejected.
* **Rule 2: No Borrowing.** Margin borrowing is disabled. *(Balance checking before orders is planned but not yet implemented.)*
* **Rule 3: Inventory Management.** *(Stop-loss logic is planned but not yet implemented.)*

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Execution Engine:** `ccxt` (configured for Spot API)
* **Message Broker:** Redis
* **State Management:** SQLite (Local)
* **Analysis:** `pandas`, `pandas_ta` (ADX/ATR calculation)
* **Dashboard:** `Streamlit`
* **Config:** `python-dotenv` for `.env` support

## 📂 Project Structure

```bash
spot-grid-swarm/
├── manager/
│   └── orchestrator.py    # Main process; manages worker lifecycle
├── workers/
│   ├── grid_bot.py        # Individual worker logic
│   └── order_manager.py   # CCXT wrapper with strict Spot-only rules
├── shared/
│   ├── config.py          # .env + ${VAR} config resolution and coercion
│   ├── messaging.py       # Redis class wrappers
│   └── database.py        # SQLite interface
├── config/
│   ├── settings.yaml      # Exchange keys and system constants
│   └── strategies.json    # Grid parameters (Upper/Lower limits) per pair
├── tests/                 # Smoke and unit tests
├── dashboard/             # Streamlit UI
├── .agent/workflows/      # Agentic runbooks (setup/start)
├── requirements.txt
└── README.md

```

## 🚀 Usage

### 1. Prerequisites

Ensure **Redis** is installed and running.

```bash
redis-server
```

**Install Dependencies (recommended in a venv):**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and set your values:

```bash
cp .env.example .env
```

The system resolves `${VAR}` or `${VAR:-default}` in `config/settings.yaml` via `python-dotenv`.
Numeric values (e.g., `REDIS_PORT`, `RISK_PER_BOT`) are coerced to `int/float` with validation.

Example `config/settings.yaml`:

```yaml
exchange:
  name: binance
  mode: testnet  # Toggle 'live' for production
  
  # Key Pool (Required for Multiple Workers)
  pool:
    - api_key: "${BINANCE_API_KEY_1}"
      secret: "${BINANCE_SECRET_KEY_1}"
    - api_key: "${BINANCE_API_KEY_2}"
      secret: "${BINANCE_SECRET_KEY_2}"

swarm:
  max_concurrency: 5
  base_currency: USDT
  risk_per_bot: 100.0  # Allocation in quote currency

```

Update `config/strategies.json` with your pair settings. The worker will exit if the pair is missing or disabled.

### 3. Deployment

**Start the Manager:**

```bash
python manager/orchestrator.py

```

**Spawn a Worker:**

```bash
python workers/grid_bot.py --pair SOL/USDT --grids 20

```

**Launch Dashboard:**

```bash
streamlit run dashboard/app.py

```

### 4. Backtesting

**Run basic backtest (Fixed Grid):**
```bash
python -m backtest.runner --pair SOL/USDT --days 30 --grids 20
```

**Run Infinity Grid (Rolling) backtest:**
```bash
python -m backtest.runner --pair SOL/USDT --days 30 --grids 20 --rolling
```

**Run Portfolio Backtest (Aggregated):**
```bash
python -m backtest.portfolio_runner --days 30 --capital 1000
```

### 5. Workflows (Agentic Mode)

This project includes pre-defined workflows for agentic IDEs in `.agent/workflows`:

*   **Setup:** `setup.md` - Installs dependencies and checks configuration.
*   **Start Manager:** `start_manager.md` - Launches the Orchestrator.
*   **Start Worker:** `start_worker.md` - Spawns a grid bot (default SOL/USDT).
*   **Start Dashboard:** `start_dashboard.md` - Starts the control room.

### 6. Tests

```bash
pytest -q
python tests/verify_env.py
```

## 🔒 Security Notes

* The dashboard redacts config secrets (keys/tokens) before display.
* `.env` is gitignored by default.

## ⚠️ Risk Disclaimer

*   **Market Risk:** This software automates trading. In trending bearish markets, grid strategies may result in unrealized losses (holding assets while price drops).
*   **Software Status:** This is Alpha-grade engineering software. Thorough backtesting and Testnet validation are required before live deployment.
*   **No Warranty:** The software is provided "as is", without warranty of any kind.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed architecture change.
