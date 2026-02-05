# 🏗 Architecture Roadmap & Known Risks

This document outlines critical architectural challenges for the **Spot-Grid-Swarm** distributed system, referencing how established open-source projects (Freqtrade, Hummingbot) handle them, and the recommended solutions for our implementation.

## 🚨 Critical Priority (Must Fix Before Live)

### 1. Nonce Race Conditions (Redis Swarm Pattern)
**The Risk:**
In a distributed "Swarm" where multiple worker processes share the same Exchange API Key, they will generate conflicting `nonces` (timestamps). If Worker A sends a request with `nonce=100` and Worker B sends `nonce=100` (or `99`) milliseconds later, the exchange will reject Worker B with `InvalidNonce`.

**How Others Solve It:**
*   **Freqtrade:** Enforces a **Strict Monolithic Process**. One bot instance = One API Key. They do not allow multiple processes to share a key.
*   **CCXT:** Explicitly advises against sharing a single key across threads/processes without a semaphore.
    *   *Ref:* [CCXT Manual - Concurrency](https://github.com/ccxt/ccxt/wiki/Manual#concurrency)

**✅ Solution for Spot-Grid-Swarm:**
- [x] **Option A (Easier):** Use **Sub-Accounts**. Implemented as a "Dynamic Key Pool" in `settings.yaml`. Workers acquire a unique key lock from Redis at startup.
- [ ] **Option B (Harder):** **Centralized Gatekeeper**. (Discarded in favor of Option A).

---

### 2. Distributed Rate Limits (The "Thundering Herd")
**The Risk:**
Exchange rate limits are Global (per IP or per Account). 50 Workers checking prices every 1 second = 50 requests/sec. This will trigger an IP Ban (usually ~10-20 req/sec limit for public endpoints).

**How Others Solve It:**
*   **Hummingbot:** Uses a complex `RateLimitCoordinator` class that tracks global usage tokens.
*   **Freqtrade:** "Downsizes" the bot to run slower than the limit using static configuration.

**✅ Solution for Spot-Grid-Swarm:**
- [ ] **Global Token Bucket:** Implement a Rate Limiter in Redis (e.g., `INCR` valid_requests). Workers check Redis before calling CCXT.
- [ ] **Shared Market Data:** Workers should **NOT** poll prices individually. The **Manager** should poll prices once and broadcast updates via Redis Pub/Sub (`market_data:SOL/USDT`). Workers simply listen.

---

## ⚠️ High Reliability (Production Hardening)

### 3. Redis Pub/Sub Data Loss
**The Risk:**
Redis Pub/Sub is "Fire and Forget". If a Worker crashes or disconnects for 1 second, it will **permanently miss** any "STOP" or "RECONFIG" commands sent during that down time.

**How Others Solve It:**
*   **Redis Best Practices:** Use **Redis Streams** (`XADD`, `XREADGROUP`) for valid message persistence.
*   **Hummingbot:** Uses a hybrid approach (WebSockets for speed, REST polling for reconciliation).

**✅ Solution for Spot-Grid-Swarm:**
- [ ] **Migrate Critical Commands:** Move `STOP`, `START`, `PANIC` commands from Pub/Sub to **Redis Streams**. This ensures Workers process them upon reconnection.
- [ ] **Keep Tickers on Pub/Sub:** Market data is ephemeral; it's okay to miss an old price tick.

### 4. WebSocket "Silent Death"
**The Risk:**
WebSockets often disconnect "silently" (the TCP connection remains open, but no data flows). A Grid Bot waiting for a price update might hang indefinitely while the market crashes.

**How Others Solve It:**
*   **Hummingbot:** Implements a strict `Heartbeat`. If no message is received for `N` seconds, it forces a reconnection.

**✅ Solution for Spot-Grid-Swarm:**
- [ ] **Watchdog Timer:** In `grid_bot.py`, implement a background thread that checks:
    ```python
    if (time.now() - last_price_update) > 15_seconds:
        raise ConnectionError("Stale Data - Reconnecting")
    ```

---

## 📉 Optimization (Future Scale)

### 5. State Drift (Local vs Exchange)
**The Risk:**
The local database (SQLite) says we have 10 Buy Orders open. The Exchange executed 5 of them, but the network packet was lost. The bot thinks it still has 10 orders.

**How Others Solve It:**
*   **Freqtrade:** Runs a `reconcile()` loop every few minutes that downloads *all* open orders from the exchange and overwrites the local state.

**✅ Solution for Spot-Grid-Swarm:**
- [ ] **Reconciliation Loop:** Add a `sync_state()` method in the Worker that runs on boot and periodically (e.g., every 5 mins) to align executed grids.
