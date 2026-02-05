# Codebase Review Report (2026-02-05)

## Updates Read Before Writing This Report
1. Reviewed recent history through `7f6e385` (`git log --oneline --decorate -n 12`).
2. Reviewed current working-tree changes (`git status --short`, `git diff --name-only`).
3. Reviewed the only local diff in `config/strategies.json` (SOL/ETH grid parameter updates).
4. Re-ran test baseline: `venv/bin/python -m pytest -q` -> `40 passed in 2.46s`.

## Verification Summary
1. Confirmed findings: 13
2. Partially confirmed findings (refined framing): 5
3. Retracted findings: 0

## Confirmed Findings
1. `Critical` Infinite blocking risk in command stream read.
Evidence: `workers/grid_bot.py:490`, `shared/messaging.py:127`
Detail: `xreadgroup(..., block=0)` is treated as non-blocking in comments, but `BLOCK 0` blocks indefinitely.

2. `High` Stale-data watchdog ineffective in live loop ordering.
Evidence: `workers/grid_bot.py:457`, `workers/grid_bot.py:462`
Detail: `last_price_update` is set immediately after fetch, then stale check runs, so age is near-zero on successful fetches.

3. `High` API key lock renewal does not verify owner.
Evidence: `workers/grid_bot.py:133`
Detail: lock TTL is extended without checking key value still matches this worker ID.

4. `High` Worker lifecycle leak can exhaust concurrency permanently.
Evidence: `manager/risk_engine.py:17`
Detail: active workers are added but never removed/expired.

5. `High` Backtest same-candle fill ordering can bias results.
Evidence: `backtest/simulator.py:165`, `backtest/simulator.py:205`
Detail: buy fills are processed before sell fills in the same candle, allowing optimistic intrabar assumptions.

6. `Medium` DB connection leak risk on error paths.
Evidence: `shared/database.py:50`, `shared/database.py:72`
Detail: connections are closed on success paths, but not guaranteed in exception paths.

7. `Medium` Rate limiter fails open on Redis errors.
Evidence: `shared/rate_limiter.py:81`
Detail: exceptions in limiter return allow (`True`), increasing external API limit risk during outages.

8. `Medium` Repeated STOP broadcasts on persistent risk breach.
Evidence: `manager/orchestrator.py:97`
Detail: STOP command is sent every loop iteration without debounce/latch.

9. `Medium` Dashboard may trust conflicting `worker_id` in payload.
Evidence: `dashboard/state.py:53`
Detail: payload `worker_id` can override canonical Redis hash key ID.

10. `Medium` Backtest cache sufficiency assumes hourly candles.
Evidence: `backtest/data_fetcher.py:82`
Detail: `days * 24` threshold ignores configured timeframe.

11. `Medium` Sharpe annualization hardcoded to hourly frequency.
Evidence: `backtest/metrics.py:92`
Detail: fixed `8760` periods per year misstates Sharpe for non-hourly data.

12. `Medium` Profit factor handling is incorrect when no losing trades exist.
Evidence: `backtest/metrics.py:70`
Detail: forcing `gross_loss = 1.0` yields finite PF instead of explicit no-loss behavior.

13. `Medium` CLI override logic ignores valid zero-like values.
Evidence: `backtest/runner.py:148`
Detail: `arg or default` pattern overrides explicit `0` / `0.0`.

## Partially Confirmed / Refined Findings
1. Silent publish-success risk is confirmed for manager path only.
Evidence: `manager/orchestrator.py:119`, `shared/messaging.py:14`, `dashboard/state.py:17`
Refinement: dashboard path already checks subscriber count semantics separately.

2. Precision/float risk is real but impact is exchange-adapter dependent.
Evidence: `workers/grid_bot.py:330`, `workers/order_manager.py:29`
Refinement: no explicit precision normalization is visible in this path; runtime effect may vary by exchange/CCXT behavior.

3. Infinite retry behavior applies to runtime loop failures, not all startup failures.
Evidence: `manager/orchestrator.py:87`
Refinement: `run()` loop catches/retries runtime exceptions; constructor/load-time failures are outside this loop.

4. Test-gap claim needed narrowing.
Evidence: `tests/test_dashboard_state.py:24`, `tests/test_risk_integration.py:61`
Refinement: there are existing failure-path tests, but targeted gaps remain for manager publish semantics, worker cleanup lifecycle, STOP debounce, and `worker_id` mismatch handling.

5. README stop-loss statement is outdated at system level.
Evidence: `readme.md:44`, `workers/grid_bot.py:515`, `tests/test_stop_loss.py:45`
Refinement: stop-loss exists in `GridBot` flow; README statement is stale.

## Priority Fix Order
1. Fix stream blocking semantics in command reads.
2. Correct lock renewal to use ownership-checked renew logic.
3. Add worker expiry/unregister path for `active_bots`.
4. Correct backtest candle fill assumptions and metrics time-scaling.
5. Add targeted regression tests for the confirmed high-impact cases.
