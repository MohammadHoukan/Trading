
import logging
from collections import defaultdict

class RiskEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("RiskEngine")
        
        # Limits from config or defaults
        self.max_capital_per_bot = config['swarm'].get('risk_per_bot', 100.0)
        self.max_global_capital = config['swarm'].get('max_global_capital', 1000.0)
        self.max_concurrency = config['swarm'].get('max_concurrency', 5)
        
        # State
        self.allocations = defaultdict(float) # worker_id -> amount
        self.active_bots = set()

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
            "allocations": dict(self.allocations)
        }
