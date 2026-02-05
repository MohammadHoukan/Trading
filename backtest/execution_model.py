"""
Execution Model for Realistic Backtesting.

Models real-world execution effects:
- Slippage: Price moves against you during fill
- Partial Fills: Not all orders get fully executed
- Spread: Bid-ask difference affects entry/exit prices
"""

import random
import logging

logger = logging.getLogger("ExecutionModel")


class ExecutionModel:
    """
    Simulates realistic order execution for backtesting.
    
    Default parameters are conservative estimates for major crypto pairs.
    """
    
    def __init__(
        self,
        slippage_bps: float = 5.0,      # 5 basis points (0.05%) average slippage
        fill_probability: float = 0.6,   # 60% of limit orders get filled
        spread_bps: float = 10.0,        # 10 basis points bid-ask spread
        partial_fill_min: float = 0.3,   # Minimum 30% fill when partially filled
        enabled: bool = True
    ):
        """
        Initialize execution model.
        
        Args:
            slippage_bps: Average slippage in basis points (100 bps = 1%)
            fill_probability: Probability that a limit order gets filled (0-1)
            spread_bps: Bid-ask spread in basis points
            partial_fill_min: Minimum fill percentage when order is partially filled
            enabled: If False, model behaves like ideal execution (100% fill, no slip)
        """
        self.slippage_bps = slippage_bps
        self.fill_probability = fill_probability
        self.spread_bps = spread_bps
        self.partial_fill_min = partial_fill_min
        self.enabled = enabled
        
        logger.info(
            f"ExecutionModel initialized: enabled={enabled}, "
            f"slippage={slippage_bps}bps, fill_prob={fill_probability}, spread={spread_bps}bps"
        )

    def simulate_fill(
        self,
        side: str,
        limit_price: float,
        amount: float,
        market_price: float,
        volume: float = None
    ) -> tuple[float, float, bool]:
        """
        Simulate order execution with realistic effects.
        
        Args:
            side: 'buy' or 'sell'
            limit_price: The limit order price
            amount: Order amount
            market_price: Current market mid-price
            volume: Recent volume (optional, affects fill probability)
            
        Returns:
            Tuple of (filled_amount, fill_price, was_filled)
        """
        if not self.enabled:
            # Ideal execution: 100% fill at limit price
            return amount, limit_price, True
        
        # Step 1: Determine if order gets filled at all
        fill_prob = self._adjust_fill_probability(limit_price, market_price, side)
        
        if random.random() > fill_prob:
            # Order not filled
            return 0.0, 0.0, False
        
        # Step 2: Determine fill amount (partial fills)
        fill_pct = self._calculate_fill_percentage()
        filled_amount = amount * fill_pct
        
        # Step 3: Calculate fill price with slippage
        fill_price = self._apply_slippage(limit_price, side)
        
        # Step 4: Apply spread
        fill_price = self._apply_spread(fill_price, side)
        
        return filled_amount, fill_price, True

    def _adjust_fill_probability(
        self,
        limit_price: float,
        market_price: float,
        side: str
    ) -> float:
        """
        Adjust fill probability based on order aggressiveness.
        
        Orders closer to market price are more likely to fill.
        """
        base_prob = self.fill_probability
        
        # Calculate distance from market price
        if side == 'buy':
            # Buy orders: higher price = more aggressive = higher fill rate
            distance_pct = (market_price - limit_price) / market_price
        else:
            # Sell orders: lower price = more aggressive = higher fill rate
            distance_pct = (limit_price - market_price) / market_price
        
        # Adjust probability: closer to market = higher prob
        # Distance of 0% = base_prob * 1.5
        # Distance of 2% = base_prob * 0.5
        adjustment = 1.5 - (distance_pct * 50)  # Steeper penalty for distant orders
        adjustment = max(0.3, min(1.5, adjustment))  # Clamp between 0.3 and 1.5
        
        return min(1.0, base_prob * adjustment)

    def _calculate_fill_percentage(self) -> float:
        """
        Determine what percentage of the order gets filled.
        
        Uses beta distribution to simulate realistic partial fills.
        """
        # 70% chance of full fill, 30% chance of partial
        if random.random() > 0.3:
            return 1.0
        
        # Partial fill: random between min and 100%
        return random.uniform(self.partial_fill_min, 1.0)

    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to the fill price.
        
        Slippage always works against the trader.
        """
        # Slippage follows exponential distribution (most are small, some large)
        slip_pct = random.expovariate(1.0 / (self.slippage_bps / 10000))
        slip_pct = min(slip_pct, 0.01)  # Cap at 1% for sanity
        
        if side == 'buy':
            return price * (1 + slip_pct)  # Buy at higher price
        else:
            return price * (1 - slip_pct)  # Sell at lower price

    def _apply_spread(self, price: float, side: str) -> float:
        """
        Apply bid-ask spread to the fill price.
        """
        spread_pct = self.spread_bps / 10000 / 2  # Half spread each direction
        
        if side == 'buy':
            return price * (1 + spread_pct)  # Buy at ask
        else:
            return price * (1 - spread_pct)  # Sell at bid

    def get_effective_costs(self) -> dict:
        """
        Return expected execution costs for analysis.
        """
        expected_slip = self.slippage_bps / 10000
        expected_spread = self.spread_bps / 10000 / 2
        expected_fill_loss = 1 - self.fill_probability
        
        return {
            'slippage_pct': expected_slip * 100,
            'spread_pct': expected_spread * 100,
            'missed_fills_pct': expected_fill_loss * 100,
            'total_expected_drag_pct': (expected_slip + expected_spread) * 100
        }
