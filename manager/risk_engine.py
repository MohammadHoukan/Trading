
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from enum import Enum


class DrawdownAction(Enum):
    """Actions based on drawdown severity."""
    NORMAL = "NORMAL"           # No action needed
    REDUCE_EXPOSURE = "REDUCE"  # Scale down position sizes
    HALT_ALL = "HALT"           # Stop all trading


@dataclass
class EquitySnapshot:
    """Point-in-time equity snapshot."""
    timestamp: float
    total_exposure: float       # Sum of all worker exposures (inventory * price)
    realized_pnl: float         # Sum of all realized profits
    unrealized_pnl: float       # Current mark-to-market PnL
    equity: float               # realized_pnl + unrealized_pnl (relative to initial)


class RiskEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("RiskEngine")

        # Limits from config or defaults
        self.max_capital_per_bot = config['swarm'].get('risk_per_bot', 100.0)
        self.max_global_capital = config['swarm'].get('max_global_capital', 1000.0)
        self.max_concurrency = config['swarm'].get('max_concurrency', 5)

        # Drawdown thresholds from config
        drawdown_cfg = config.get('risk', {}).get('drawdown', {})
        self.drawdown_warning_pct = drawdown_cfg.get('warning_pct', 10.0)    # 10%
        self.drawdown_reduce_pct = drawdown_cfg.get('reduce_pct', 15.0)      # 15%
        self.drawdown_halt_pct = drawdown_cfg.get('halt_pct', 20.0)          # 20%
        self.position_scale_factor = drawdown_cfg.get('scale_factor', 0.5)   # 50% reduction

        # State - allocations and workers
        self.allocations = defaultdict(float)  # worker_id -> exposure amount
        self.active_bots = set()

        # State - equity tracking
        self.worker_pnl: Dict[str, float] = {}           # worker_id -> realized_pnl
        self.worker_unrealized: Dict[str, float] = {}    # worker_id -> unrealized_pnl
        self.peak_equity: float = 0.0                    # High water mark
        self.current_equity: float = 0.0                 # Current equity
        self.last_equity_update: float = 0.0
        self.drawdown_action: DrawdownAction = DrawdownAction.NORMAL

    def register_worker(self, worker_id, symbol):
        """Register a new worker."""
        if len(self.active_bots) >= self.max_concurrency:
            if worker_id not in self.active_bots:
                self.logger.warning(f"Registration REJECTED for {worker_id}: Max concurrency reached.")
                return False
        
        self.active_bots.add(worker_id)
        self.logger.info(f"Worker {worker_id} ({symbol}) registered.")
        return True

    def unregister_worker(self, worker_id):
        """Remove a worker and clear its allocation."""
        if worker_id in self.active_bots:
            self.active_bots.remove(worker_id)
        
        if worker_id in self.allocations:
            reserved = self.allocations.pop(worker_id)
            self.logger.info(f"Worker {worker_id} unregistered. Released {reserved:.2f} capital.")

    def request_allocation(self, worker_id, amount):
        """Request capital allocation for a trade."""
        current_alloc = self.allocations[worker_id]
        new_alloc = current_alloc + amount
        
        # 1. Per-Bot Check
        if new_alloc > self.max_capital_per_bot:
            self.logger.warning(f"Risk Reject {worker_id}: Request {amount} exceeds limit {self.max_capital_per_bot}")
            return False
            
        # 2. Global Check
        total_alloc = sum(self.allocations.values())
        if (total_alloc + amount) > self.max_global_capital:
             self.logger.warning(f"Risk Reject {worker_id}: Global limit {self.max_global_capital} reached.")
             return False
             
        self.allocations[worker_id] = new_alloc
        return True

    def update_exposure(self, worker_id, exposure):
        """Update exposure based on worker report."""
        self.allocations[worker_id] = exposure

    def release_allocation(self, worker_id, amount):
        """Release capital (e.g. after a sell)."""
        self.allocations[worker_id] = max(0.0, self.allocations[worker_id] - amount)

    def get_status(self):
        return {
            "total_allocated": sum(self.allocations.values()),
            "active_bots": list(self.active_bots),
            "allocations": dict(self.allocations),
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self.get_drawdown_pct(),
            "drawdown_action": self.drawdown_action.value,
        }

    # ==================== EQUITY & DRAWDOWN TRACKING ====================

    def update_worker_pnl(self, worker_id: str, realized_pnl: float, unrealized_pnl: float):
        """
        Update PnL for a worker (called from orchestrator on worker status updates).

        Args:
            worker_id: Worker identifier
            realized_pnl: Total realized profit/loss for this worker
            unrealized_pnl: Current unrealized PnL (inventory * price - inventory * avg_cost)
        """
        self.worker_pnl[worker_id] = realized_pnl
        self.worker_unrealized[worker_id] = unrealized_pnl
        self._recalculate_equity()

    def _recalculate_equity(self):
        """Recalculate total equity and update high water mark."""
        total_realized = sum(self.worker_pnl.values())
        total_unrealized = sum(self.worker_unrealized.values())

        self.current_equity = total_realized + total_unrealized

        # Update high water mark
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self.logger.debug(f"New peak equity: ${self.peak_equity:.2f}")

    def get_drawdown_pct(self) -> float:
        """
        Calculate current drawdown percentage from peak.

        Returns:
            Drawdown as a positive percentage (e.g., 15.0 for 15% drawdown)
        """
        if self.peak_equity <= 0:
            return 0.0

        drawdown = self.peak_equity - self.current_equity
        return (drawdown / self.peak_equity) * 100 if drawdown > 0 else 0.0

    def check_drawdown(self) -> DrawdownAction:
        """
        Check drawdown and determine required action.

        Returns:
            DrawdownAction indicating what action to take
        """
        drawdown_pct = self.get_drawdown_pct()

        if drawdown_pct >= self.drawdown_halt_pct:
            if self.drawdown_action != DrawdownAction.HALT_ALL:
                self.logger.critical(
                    f"DRAWDOWN HALT: {drawdown_pct:.1f}% exceeds {self.drawdown_halt_pct}% threshold! "
                    f"Equity: ${self.current_equity:.2f}, Peak: ${self.peak_equity:.2f}"
                )
            self.drawdown_action = DrawdownAction.HALT_ALL
            return DrawdownAction.HALT_ALL

        elif drawdown_pct >= self.drawdown_reduce_pct:
            if self.drawdown_action != DrawdownAction.REDUCE_EXPOSURE:
                self.logger.warning(
                    f"DRAWDOWN REDUCE: {drawdown_pct:.1f}% exceeds {self.drawdown_reduce_pct}% threshold. "
                    f"Scaling positions to {self.position_scale_factor*100:.0f}%"
                )
            self.drawdown_action = DrawdownAction.REDUCE_EXPOSURE
            return DrawdownAction.REDUCE_EXPOSURE

        elif drawdown_pct >= self.drawdown_warning_pct:
            if self.drawdown_action == DrawdownAction.NORMAL:
                self.logger.warning(
                    f"DRAWDOWN WARNING: {drawdown_pct:.1f}% approaching limit. "
                    f"Equity: ${self.current_equity:.2f}, Peak: ${self.peak_equity:.2f}"
                )
            # Don't change action, just warn
            return self.drawdown_action

        else:
            # Recovery - reset to normal if we were in a reduced state
            if self.drawdown_action != DrawdownAction.NORMAL:
                self.logger.info(
                    f"Drawdown recovered to {drawdown_pct:.1f}%. Resuming normal operations."
                )
            self.drawdown_action = DrawdownAction.NORMAL
            return DrawdownAction.NORMAL

    def get_position_scale(self) -> float:
        """
        Get position scaling factor based on current drawdown state.

        Returns:
            Multiplier for position sizes (1.0 = normal, 0.5 = reduced)
        """
        if self.drawdown_action == DrawdownAction.REDUCE_EXPOSURE:
            return self.position_scale_factor
        elif self.drawdown_action == DrawdownAction.HALT_ALL:
            return 0.0
        return 1.0

    def reset_peak_equity(self, new_peak: float = None):
        """
        Reset peak equity (e.g., after depositing more capital).

        Args:
            new_peak: New peak value. If None, uses current equity.
        """
        if new_peak is not None:
            self.peak_equity = new_peak
        else:
            self.peak_equity = self.current_equity
        self.drawdown_action = DrawdownAction.NORMAL
        self.logger.info(f"Peak equity reset to ${self.peak_equity:.2f}")

    def cleanup_worker_pnl(self, worker_id: str):
        """Remove PnL tracking for a terminated worker."""
        self.worker_pnl.pop(worker_id, None)
        self.worker_unrealized.pop(worker_id, None)
        self._recalculate_equity()
