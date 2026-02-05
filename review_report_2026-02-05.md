# Codebase Review Report (Updated 2026-02-05)

## Scope
1. Re-reviewed current `HEAD` after latest commits (`46a1f92` and predecessors).
2. Focused on behavior changes in `workers/`, `manager/`, `shared/`, and `backtest/`.
3. Validation run: `venv/bin/python -m pytest -q` -> `40 passed in 2.37s`.

## Findings (Current State)
1. `High` Global STOP can be silently dropped after a transient publish failure.
Evidence: `manager/orchestrator.py:107`, `manager/orchestrator.py:134`, `manager/orchestrator.py:137`
Detail: `stop_broadcast_sent` is set after attempting STOP broadcast, but `broadcast_command` does not return success/failure to gate that flag. A single failed publish can suppress further STOP attempts while breach persists.

2. `High` Worker unregistration path is mostly unreachable from worker shutdown paths.
Evidence: `manager/orchestrator.py:42`, `workers/grid_bot.py:560`, `workers/grid_bot.py:549`
Detail: manager only unregisters on terminal status updates, but worker STOP/STOP_LOSS paths do not publish terminal status to the manager status channel. This can leave stale `active_bots` and allocations.

3. `High` Profit factor regression for break-even strategies (`0/0 -> inf`).
Evidence: `backtest/metrics.py:69`, `backtest/metrics.py:70`, `backtest/metrics.py:71`
Detail: when all sell-trade PnL values are zero, winners and losers are both empty and `profit_factor` becomes `inf`, which is mathematically misleading for break-even performance.

4. `Medium` Cache validation can accept severely undersampled history.
Evidence: `backtest/data_fetcher.py:84`, `backtest/data_fetcher.py:85`
Detail: cache completeness is validated only by time span (`last - first`) and not by candle density/count, so sparse data can be accepted as full coverage.

5. `Medium` Terminal worker states are not persisted to dashboard snapshot store.
Evidence: `manager/orchestrator.py:42`, `manager/orchestrator.py:45`, `manager/orchestrator.py:64`
Detail: terminal updates return early before writing `workers:data`, so dashboard may keep showing old `RUNNING/PAUSED` state until stale timeout.

6. `Medium` API key lock-loss detection is not fail-safe.
Evidence: `workers/grid_bot.py:144`, `workers/grid_bot.py:145`
Detail: on lock ownership loss, renewal stops but bot continues running with same key. This can allow concurrent key usage if another worker acquires the lock.

7. `Low` Test name drift in rate limiter suite.
Evidence: `tests/test_rate_limiter.py:65`
Detail: function name still says `fail_open`, but behavior and assertion are now fail-closed.

## Coverage Gaps
1. No test for STOP retry semantics when publish fails under sustained global-risk breach.
Evidence: `manager/orchestrator.py:107`, `tests/test_risk_integration.py:57`

2. No test that worker terminal statuses are published and trigger unregistration end-to-end.
Evidence: `manager/orchestrator.py:42`, `workers/grid_bot.py:560`, `tests/test_risk_integration.py:62`

3. No unit tests for backtest edge cases introduced by metric/cache updates.
Evidence: `backtest/metrics.py:69`, `backtest/data_fetcher.py:84`

4. No deterministic tests for intra-candle phase ordering behavior in simulator.
Evidence: `backtest/simulator.py:165`

## Resolved Since Prior Report
1. Stream command read now uses bounded blocking instead of indefinite block.
Evidence: `workers/grid_bot.py:503`

2. Lock renewal now verifies ownership before extending TTL.
Evidence: `workers/grid_bot.py:136`, `workers/grid_bot.py:142`

3. Database write paths now close connections in `finally`.
Evidence: `shared/database.py:51`, `shared/database.py:75`

4. Rate limiter now fails closed on Redis errors.
Evidence: `shared/rate_limiter.py:83`

5. Dashboard worker identity now uses Redis hash key as source of truth.
Evidence: `dashboard/state.py:54`, `dashboard/state.py:58`

## Priority Fix Order
1. Gate `stop_broadcast_sent` on successful STOP publish, and retry while breach persists.
2. Publish terminal status from workers on STOP/STOP_LOSS/ERROR so manager can unregister reliably.
3. Fix `profit_factor` semantics for break-even/no-loss edge cases.
4. Strengthen cache validation with expected-candle-density checks.
5. Add regression tests for the high-impact runtime and backtest edge cases above.
