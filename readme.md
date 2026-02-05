# 🌐 Spot-Grid-Swarm

**A Distributed, Multi-Agent Trading System for Spot Market Grid Strategies.**

## 📖 Overview

**Spot-Grid-Swarm** is a Python-based trading architecture designed to orchestrate a cluster of independent grid trading bots.

This system is engineered with strict **Risk & Asset Constraints**:

1. **Spot Market Only:** No interaction with Futures, Options, or Derivatives.
2. **Zero Leverage:** Operates strictly on a 1:1 capital basis (no margin borrowing).
3. **Long-Only:** No short selling. The system focuses on accumulating and selling the underlying asset.

It utilizes a **Hub-and-Spoke** architecture to manage concurrency, allowing multiple assets to be traded simultaneously while maintaining a global risk state.

## 🏗 Architecture

The system mimics a microservices pattern using **Redis Pub/Sub** for inter-process communication:

* **👑 The Manager (Orchestrator):**
* Monitors global market volatility (Regime Detection).
* Allocates capital dynamically to workers.
* Enforces global "Kill Switch" protocols.


* **🐝 The Workers (Swarm):**
* Independent processes spawned per trading pair (e.g., `SOL/USDT`, `ETH/USDT`).
* Execute localized Grid Trading logic (Buy Low / Sell High).
* Maintain their own order state and heartbeat.


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

## 📂 Project Structure

```bash
spot-grid-swarm/
├── manager/
│   ├── orchestrator.py    # Main process; manages worker lifecycle
│   ├── regime_filter.py   # [STUB] ADX/ATR logic for market regime detection
│   └── risk_engine.py     # [STUB] Enforces global exposure limits
├── workers/
│   ├── grid_bot.py        # Individual worker logic
│   └── order_manager.py   # CCXT wrapper with strict Spot-only rules
├── shared/
│   ├── messaging.py       # Redis class wrappers
│   └── database.py        # SQLite interface
├── config/
│   ├── settings.yaml      # Exchange keys and system constants
│   └── strategies.json    # [STUB] Grid parameters (Upper/Lower limits) per pair
├── tests/                 # Smoke and unit tests
├── dashboard/             # Streamlit UI
├── requirements.txt
└── README.md

```

## 🚀 Usage

### 1. Prerequisites

Ensure **Redis** is installed and running.

```bash
redis-server
```

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/settings.yaml`:

```yaml
exchange:
  name: binance
  mode: testnet  # Toggle 'live' for production
  api_key: "YOUR_KEY"
  secret: "YOUR_SECRET"

swarm:
  max_concurrency: 5
  base_currency: USDT
  risk_per_bot: 100.0  # Allocation in quote currency

```

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

### 4. Workflows (Agentic Mode)

This project includes pre-defined workflows for agentic IDEs in `.agent/workflows`:

*   **Setup:** `setup.md` - Installs dependencies and checks configuration.
*   **Start Manager:** `start_manager.md` - Launches the Orchestrator.
*   **Start Worker:** `start_worker.md` - Spawns a grid bot (default SOL/USDT).
*   **Start Dashboard:** `start_dashboard.md` - Starts the control room.

## ⚠️ Risk Disclaimer

*   **Market Risk:** This software automates trading. In trending bearish markets, grid strategies may result in unrealized losses (holding assets while price drops).
*   **Software Status:** This is Alpha-grade engineering software. Thorough backtesting and Testnet validation are required before live deployment.
*   **No Warranty:** The software is provided "as is", without warranty of any kind.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed architecture change.