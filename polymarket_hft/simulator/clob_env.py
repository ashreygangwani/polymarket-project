"""
Polymarket CLOB Simulator — Phase 2 Simulation Environment.

Simulates the Polymarket Central Limit Order Book with:
    - March 2026 fee curve (1.56% taker fee at 50/50, 20% maker rebate)
    - Fill simulation: resting maker orders filled when price crosses
    - State tracking: USDC balance, YES/NO shares, unrealized PnL
    - Position lifecycle: entry, scale-out, runner management

This is a deterministic, event-driven simulator — no randomness.
Every fill decision is based on the replayed price crossing our order level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("clob_sim")


# ---------------------------------------------------------------------------
# Fee Curve — Polymarket March 2026
# ---------------------------------------------------------------------------
# Taker fee peaks at ~1.56% at 50/50 odds (probability = 0.50).
# Fee scales linearly toward 0% at probabilities 0.0 and 1.0.
# Maker rebate = 20% of the collected taker fee on the matched order.

def taker_fee_rate(probability: float) -> float:
    """
    Compute the taker fee rate for a given probability.

    Polymarket 2026: fee = 2 * PEAK_RATE * min(p, 1-p)
    At p=0.50: fee = 2 * 0.0156 * 0.50 = 0.0156 (1.56%)
    At p=0.10: fee = 2 * 0.0156 * 0.10 = 0.00312 (0.312%)
    At p=0.90: fee = 2 * 0.0156 * 0.10 = 0.00312 (0.312%)
    """
    PEAK_RATE = 0.0156
    return 2.0 * PEAK_RATE * min(probability, 1.0 - probability)


def maker_rebate_rate(probability: float) -> float:
    """Maker rebate = 20% of the taker fee that was collected."""
    return 0.20 * taker_fee_rate(probability)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class Side(Enum):
    BID = "BID"   # We buy YES shares
    ASK = "ASK"   # We sell YES shares


class OrderType(Enum):
    MAKER = "MAKER"    # Post-only limit order (earns rebate)
    TAKER = "TAKER"    # Aggressive market/FAK order (pays fee)


@dataclass
class Order:
    """A resting limit order in the simulated book."""
    side: Side
    price: float           # Probability price (0.0 - 1.0)
    size_usdc: float       # USDC notional
    order_type: OrderType = OrderType.MAKER
    placed_at_idx: int = 0  # Candle index when placed
    filled: bool = False
    fill_price: float = 0.0
    fill_idx: int = 0


@dataclass
class Position:
    """Tracks an open directional position."""
    entry_price: float      # Probability at entry
    entry_usdc: float       # USDC spent
    shares: float           # YES shares acquired = usdc / price
    side: str = "YES"       # Always YES for simplicity
    entry_idx: int = 0
    scaled_out: bool = False
    runner_shares: float = 0.0
    runner_entry_usdc: float = 0.0


@dataclass
class TradeRecord:
    """Logged trade for the tear sheet."""
    idx: int
    action: str            # "MAKER_BID_FILL", "MAKER_ASK_FILL", "DIRECTIONAL_ENTRY", etc.
    price: float
    size_usdc: float
    fee_usdc: float        # Negative = rebate earned
    pnl_usdc: float
    balance_after: float


@dataclass
class AccountState:
    """Full account state at any point in time."""
    usdc_balance: float = 500.0
    yes_shares: float = 0.0
    no_shares: float = 0.0
    unrealized_pnl: float = 0.0
    total_volume: float = 0.0
    total_maker_rebates: float = 0.0
    total_taker_fees: float = 0.0
    total_directional_pnl: float = 0.0
    peak_balance: float = 500.0
    max_drawdown_pct: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)


class CLOBSimulator:
    """
    Simulates the Polymarket CLOB for backtesting.

    Usage:
        sim = CLOBSimulator(starting_capital=500.0)
        sim.place_maker_orders(fair_price=0.50, spread=0.015, size_usdc=25.0, idx=0)
        sim.process_price_update(new_price=0.52, idx=1)
        sim.enter_directional(probability=0.48, size_usdc=30.0, idx=5)
        ...
        sim.print_tear_sheet()
    """

    def __init__(
        self,
        starting_capital: float = 500.0,
        max_exposure_usdc: float = 75.0,
        stop_loss_pct: float = 0.02,
        inventory_limit: float = 40.0,
        max_skew_cents: float = 0.03,
        global_stop_distance: float = 0.08,
        cooldown_duration_ms: int = 0,
        macro_breaker_limit: float = -25.0,
    ) -> None:
        self.account = AccountState(usdc_balance=starting_capital)
        self.max_exposure = max_exposure_usdc
        self.stop_loss_pct = stop_loss_pct
        self.inventory_limit = inventory_limit
        self.max_skew_cents = max_skew_cents
        self.global_stop_distance = global_stop_distance
        self.cooldown_duration_ms = cooldown_duration_ms
        self.macro_breaker_limit = macro_breaker_limit
        self.last_stop_loss_time_ms: int = 0
        self.macro_breaker_tripped: bool = False
        self._resting_orders: list[Order] = []

    # ------------------------------------------------------------------
    # Maker Order Management
    # ------------------------------------------------------------------

    def place_maker_orders(
        self,
        fair_price: float,
        spread: float,
        size_usdc: float,
        idx: int,
        current_time_ms: int = 0,
    ) -> None:
        """
        Place a bid and ask ±spread from fair_price.

        Args:
            fair_price: Current synthetic fair price as probability (0-1)
            spread: Distance from fair price (e.g., 0.015 = 1.5 cents)
            size_usdc: USDC notional per side
            idx: Current candle index
            current_time_ms: Current timestamp for cooldown check
        """
        # Cancel existing resting orders.
        self._resting_orders = []

        bid_price = max(0.01, fair_price - spread)
        ask_price = min(0.99, fair_price + spread)

        # Enforce exposure cap.
        current_exposure = sum(
            p.entry_usdc for p in self.account.positions
        )
        available = max(0, self.max_exposure - current_exposure)
        effective_size = min(size_usdc, available, self.account.usdc_balance)

        if effective_size <= 0:
            return

        # Cooldown check: suppress bids during penalty box (Task 2.11).
        in_cooldown = self.is_in_cooldown(current_time_ms)

        if not in_cooldown:
            self._resting_orders.append(Order(
                side=Side.BID,
                price=bid_price,
                size_usdc=effective_size,
                order_type=OrderType.MAKER,
                placed_at_idx=idx,
            ))

        self._resting_orders.append(Order(
            side=Side.ASK,
            price=ask_price,
            size_usdc=effective_size,
            order_type=OrderType.MAKER,
            placed_at_idx=idx,
        ))

    def cancel_ask_orders(self) -> None:
        """Cancel resting ask orders (used during directional mode)."""
        self._resting_orders = [
            o for o in self._resting_orders if o.side != Side.ASK
        ]

    def cancel_all_orders(self) -> None:
        """Cancel all resting orders (kill switch)."""
        self._resting_orders = []

    def get_current_exposure(self) -> float:
        """Sum of entry USDC across all open positions."""
        return sum(p.entry_usdc for p in self.account.positions)

    # ------------------------------------------------------------------
    # Post-Stop Cooldown — Penalty Box (Task 2.11)
    # ------------------------------------------------------------------

    def is_in_cooldown(self, current_time_ms: int) -> bool:
        """
        Check if we are in the post-stop cooldown "penalty box".

        After a global stop-loss fires, the quoting engine is prohibited
        from placing new BID orders for cooldown_duration_ms (default 5 min).
        ASK orders are still permitted to unwind remaining inventory.

        Returns True if in cooldown (suppress bids), False otherwise.
        """
        if self.last_stop_loss_time_ms == 0:
            return False
        if self.cooldown_duration_ms <= 0:
            return False
        return current_time_ms < self.last_stop_loss_time_ms + self.cooldown_duration_ms

    # ------------------------------------------------------------------
    # Daily Macro Circuit Breaker (Task 2.12)
    # ------------------------------------------------------------------

    def check_macro_breaker(self, idx: int) -> bool:
        """
        Check if daily realized PnL has breached the macro circuit breaker.

        If total_directional_pnl <= macro_breaker_limit (default -$25),
        instantly cancel all orders, halt all trading, and log the event.

        Returns True if breaker just tripped (first time), False otherwise.
        """
        if self.macro_breaker_tripped:
            return False  # Already tripped — don't re-log.

        if self.account.total_directional_pnl <= self.macro_breaker_limit:
            self.macro_breaker_tripped = True
            self.cancel_all_orders()
            logger.warning(
                "MACRO BREAKER TRIPPED at idx=%d: realized PnL=$%.2f "
                "(limit=$%.2f). ALL TRADING HALTED.",
                idx, self.account.total_directional_pnl,
                self.macro_breaker_limit,
            )
            return True

        return False

    def reset_daily_pnl(self) -> None:
        """
        Reset daily PnL tracking for a new UTC day.

        Resets: macro_breaker_tripped, total_directional_pnl, cooldown.
        Preserves: balance, positions, shares, lifetime counters, drawdown.
        """
        self.macro_breaker_tripped = False
        self.account.total_directional_pnl = 0.0
        self.last_stop_loss_time_ms = 0

    # ------------------------------------------------------------------
    # Asymmetric Quoting — Inventory Skew (Task 2.8)
    # ------------------------------------------------------------------

    def place_skewed_maker_orders(
        self,
        fair_price: float,
        spread: float,
        size_usdc: float,
        idx: int,
        current_time_ms: int = 0,
    ) -> None:
        """
        Place inventory-skewed maker orders.

        Skew = (Current_Exposure / Inventory_Limit) × Max_Skew_Cents

        If LONG (holding YES shares):
            Shift both Bid and Ask DOWN by Skew.
            → Less likely to buy more YES (bid further from market).
            → More likely to sell YES (ask closer to market).

        If Current_Exposure >= Inventory_Limit:
            Pause the Bid entirely. Only quote Ask to unwind.
        """
        self._resting_orders = []

        current_exposure = self.get_current_exposure()
        has_inventory = self.account.yes_shares > 0

        # Compute skew: linear penalty proportional to inventory fill.
        skew = 0.0
        if has_inventory and self.inventory_limit > 0:
            skew = (current_exposure / self.inventory_limit) * self.max_skew_cents

        # Apply skew: shift both quotes DOWN when long.
        bid_price = max(0.01, fair_price - spread - skew)
        ask_price = min(0.99, fair_price + spread - skew)

        # If at/above inventory limit: ASK ONLY to unwind.
        if current_exposure >= self.inventory_limit:
            if has_inventory:
                self._resting_orders.append(Order(
                    side=Side.ASK,
                    price=ask_price,
                    size_usdc=size_usdc,
                    order_type=OrderType.MAKER,
                    placed_at_idx=idx,
                ))
            return

        # Quote both sides with skew applied.
        available = max(0, self.max_exposure - current_exposure)
        bid_size = min(size_usdc, available, self.account.usdc_balance)

        # Cooldown check: suppress bids during penalty box (Task 2.11).
        in_cooldown = self.is_in_cooldown(current_time_ms)

        if bid_size > 0 and not in_cooldown:
            self._resting_orders.append(Order(
                side=Side.BID,
                price=bid_price,
                size_usdc=bid_size,
                order_type=OrderType.MAKER,
                placed_at_idx=idx,
            ))

        # Always quote the ask side (we want to sell inventory).
        self._resting_orders.append(Order(
            side=Side.ASK,
            price=ask_price,
            size_usdc=size_usdc,
            order_type=OrderType.MAKER,
            placed_at_idx=idx,
        ))

    # ------------------------------------------------------------------
    # Global Portfolio Stop-Loss (Task 2.9)
    # ------------------------------------------------------------------

    def check_global_stop_loss(
        self, current_price: float, idx: int, current_time_ms: int = 0
    ) -> list[TradeRecord]:
        """
        Global stop-loss for maker inventory: if synthetic fair price drops
        more than global_stop_distance (8 cents) below a position's entry
        price, flatten that position via market sell immediately.

        Only applies to non-runner positions (runners have their own stop).
        Takes the loss and resets — cauterizes the bleed.
        """
        exits: list[TradeRecord] = []

        for pos in list(self.account.positions):
            if pos.shares <= 0:
                continue
            # Skip runners — they have their own check_runner_stops logic.
            if pos.scaled_out:
                continue

            if current_price < pos.entry_price - self.global_stop_distance:
                # Flatten via aggressive taker sell.
                shares = pos.shares
                proceeds_gross = shares * current_price
                fee = proceeds_gross * taker_fee_rate(current_price)
                proceeds_net = proceeds_gross - fee

                self.account.usdc_balance += proceeds_net
                self.account.yes_shares -= shares
                self.account.total_taker_fees += fee
                self.account.total_volume += proceeds_gross

                pnl = proceeds_net - pos.entry_usdc

                record = TradeRecord(
                    idx=idx,
                    action="GLOBAL_STOP_LOSS",
                    price=current_price,
                    size_usdc=proceeds_gross,
                    fee_usdc=fee,
                    pnl_usdc=pnl,
                    balance_after=self.account.usdc_balance,
                )
                self.account.trades.append(record)
                self.account.total_directional_pnl += pnl
                exits.append(record)

                logger.info(
                    "GLOBAL STOP: Flattened %.1f shares @ %.4f "
                    "(entry=%.4f, loss=$%.2f) idx=%d",
                    shares, current_price, pos.entry_price, pnl, idx,
                )

                pos.shares = 0.0
                pos.entry_usdc = 0.0

        # Clean up closed positions.
        self.account.positions = [
            p for p in self.account.positions if p.shares > 0
        ]
        if exits:
            self.update_drawdown(current_price)
            # Record stop time for cooldown penalty box (Task 2.11).
            self.last_stop_loss_time_ms = current_time_ms

        return exits

    # ------------------------------------------------------------------
    # Price Update Processing (Fill Simulation)
    # ------------------------------------------------------------------

    def process_price_update(
        self,
        price_low: float,
        price_high: float,
        idx: int,
    ) -> list[TradeRecord]:
        """
        Check if any resting orders would have filled given the price range.

        A maker BID fills if price_low <= bid_price (price swept down to us).
        A maker ASK fills if price_high >= ask_price (price swept up to us).

        Returns list of fill trade records.
        """
        fills: list[TradeRecord] = []

        for order in self._resting_orders:
            if order.filled:
                continue

            filled = False
            if order.side == Side.BID and price_low <= order.price:
                filled = True
            elif order.side == Side.ASK and price_high >= order.price:
                filled = True

            if filled:
                order.filled = True
                order.fill_price = order.price
                order.fill_idx = idx
                record = self._execute_maker_fill(order, idx)
                fills.append(record)

        # Remove filled orders.
        self._resting_orders = [
            o for o in self._resting_orders if not o.filled
        ]
        return fills

    def _execute_maker_fill(self, order: Order, idx: int) -> TradeRecord:
        """Process a maker order fill: update account, earn rebate."""
        rebate = order.size_usdc * maker_rebate_rate(order.price)
        pnl = 0.0  # Initialized for both branches.

        if order.side == Side.BID:
            # We bought YES shares at order.price.
            shares = order.size_usdc / order.price
            self.account.usdc_balance -= order.size_usdc
            self.account.usdc_balance += rebate
            self.account.yes_shares += shares

            # Track as a micro-position for PnL accounting.
            self.account.positions.append(Position(
                entry_price=order.price,
                entry_usdc=order.size_usdc,
                shares=shares,
                entry_idx=idx,
            ))

            action = "MAKER_BID_FILL"

        else:  # ASK
            # We sold YES shares at order.price.
            # For maker quoting, we need shares to sell.
            # If we have shares from prior bid fills, sell them.
            shares_to_sell = min(
                order.size_usdc / order.price,
                self.account.yes_shares,
            )
            if shares_to_sell <= 0:
                # No inventory to sell — skip this fill.
                return TradeRecord(
                    idx=idx,
                    action="MAKER_ASK_NO_INVENTORY",
                    price=order.price,
                    size_usdc=0,
                    fee_usdc=0,
                    pnl_usdc=0,
                    balance_after=self.account.usdc_balance,
                )

            proceeds = shares_to_sell * order.price
            self.account.usdc_balance += proceeds + rebate
            self.account.yes_shares -= shares_to_sell

            # Realize PnL from the oldest matching position.
            pnl = self._close_oldest_position(shares_to_sell, order.price)

            action = "MAKER_ASK_FILL"

        self.account.total_maker_rebates += rebate
        self.account.total_volume += order.size_usdc

        record = TradeRecord(
            idx=idx,
            action=action,
            price=order.price,
            size_usdc=order.size_usdc,
            fee_usdc=-rebate,  # Negative = money earned
            pnl_usdc=pnl if order.side == Side.ASK else 0.0,
            balance_after=self.account.usdc_balance,
        )
        self.account.trades.append(record)
        self.update_drawdown(order.price)
        return record

    # ------------------------------------------------------------------
    # Directional Entry & Exit (State B — Runner Logic)
    # ------------------------------------------------------------------

    def enter_directional(
        self,
        probability: float,
        size_usdc: float,
        idx: int,
    ) -> TradeRecord | None:
        """
        Aggressive taker buy of YES shares for directional scalp.

        Returns the trade record or None if insufficient capital/exposure.
        """
        # Enforce exposure cap.
        current_exposure = sum(
            p.entry_usdc for p in self.account.positions
        )
        available = max(0, self.max_exposure - current_exposure)
        effective_size = min(size_usdc, available, self.account.usdc_balance)

        if effective_size < 1.0:  # Minimum meaningful trade.
            return None

        fee = effective_size * taker_fee_rate(probability)
        cost = effective_size + fee
        if cost > self.account.usdc_balance:
            effective_size = self.account.usdc_balance / (1.0 + taker_fee_rate(probability))
            fee = effective_size * taker_fee_rate(probability)
            cost = effective_size + fee

        shares = effective_size / probability

        self.account.usdc_balance -= cost
        self.account.yes_shares += shares
        self.account.total_taker_fees += fee
        self.account.total_volume += effective_size

        position = Position(
            entry_price=probability,
            entry_usdc=effective_size,
            shares=shares,
            entry_idx=idx,
        )
        self.account.positions.append(position)

        record = TradeRecord(
            idx=idx,
            action="DIRECTIONAL_ENTRY",
            price=probability,
            size_usdc=effective_size,
            fee_usdc=fee,
            pnl_usdc=0.0,
            balance_after=self.account.usdc_balance,
        )
        self.account.trades.append(record)
        self.update_drawdown(probability)

        logger.debug(
            "DIRECTIONAL ENTRY: %.4f prob, $%.2f, %.1f shares @ idx %d",
            probability, effective_size, shares, idx,
        )
        return record

    def check_scale_out(
        self, current_price: float, idx: int
    ) -> list[TradeRecord]:
        """
        50% scale-out: when position has 50%+ unrealized gain,
        sell enough shares to recoup initial USDC basis + taker fees.
        """
        exits: list[TradeRecord] = []

        for pos in self.account.positions:
            if pos.scaled_out or pos.shares <= 0:
                continue

            current_value = pos.shares * current_price
            cost_basis = pos.entry_usdc
            unrealized_gain_pct = (current_value - cost_basis) / cost_basis

            if unrealized_gain_pct >= 0.50:
                # Sell enough to recoup basis + estimated taker fees.
                sell_fee_rate = taker_fee_rate(current_price)
                # shares_to_sell * price * (1 - fee_rate) = cost_basis
                shares_to_sell = cost_basis / (
                    current_price * (1.0 - sell_fee_rate)
                )
                shares_to_sell = min(shares_to_sell, pos.shares)

                proceeds_gross = shares_to_sell * current_price
                fee = proceeds_gross * sell_fee_rate
                proceeds_net = proceeds_gross - fee

                self.account.usdc_balance += proceeds_net
                self.account.yes_shares -= shares_to_sell
                self.account.total_taker_fees += fee
                self.account.total_volume += proceeds_gross
                pnl = proceeds_net - cost_basis

                pos.scaled_out = True
                pos.runner_shares = pos.shares - shares_to_sell
                pos.runner_entry_usdc = 0.0  # Basis recouped — risk-free.
                pos.entry_usdc = 0.0         # Zero cost basis for unrealized PnL calc.
                pos.shares = pos.runner_shares

                record = TradeRecord(
                    idx=idx,
                    action="SCALE_OUT_50PCT",
                    price=current_price,
                    size_usdc=proceeds_gross,
                    fee_usdc=fee,
                    pnl_usdc=pnl,
                    balance_after=self.account.usdc_balance,
                )
                self.account.trades.append(record)
                self.account.total_directional_pnl += pnl
                exits.append(record)

                logger.debug(
                    "SCALE OUT: sold %.1f shares @ %.4f, recouped $%.2f, "
                    "runner=%.1f shares",
                    shares_to_sell, current_price, proceeds_net,
                    pos.runner_shares,
                )

        self.update_drawdown(current_price)
        return exits

    def check_runner_stops(
        self, current_price: float, idx: int
    ) -> list[TradeRecord]:
        """
        Synthetic stop-loss for runners: exit if price < entry + 2%.
        Uses FAK (Fill-and-Kill) as taker.
        """
        exits: list[TradeRecord] = []

        for pos in list(self.account.positions):
            if not pos.scaled_out or pos.runner_shares <= 0:
                continue

            stop_price = pos.entry_price * (1.0 + self.stop_loss_pct)
            if current_price < stop_price:
                # Liquidate runner via taker.
                shares = pos.runner_shares
                proceeds_gross = shares * current_price
                fee = proceeds_gross * taker_fee_rate(current_price)
                proceeds_net = proceeds_gross - fee

                self.account.usdc_balance += proceeds_net
                self.account.yes_shares -= shares
                self.account.total_taker_fees += fee
                self.account.total_volume += proceeds_gross

                # Runner was risk-free (basis=0), so all proceeds are PnL.
                pnl = proceeds_net

                record = TradeRecord(
                    idx=idx,
                    action="RUNNER_STOP_EXIT",
                    price=current_price,
                    size_usdc=proceeds_gross,
                    fee_usdc=fee,
                    pnl_usdc=pnl,
                    balance_after=self.account.usdc_balance,
                )
                self.account.trades.append(record)
                self.account.total_directional_pnl += pnl
                exits.append(record)

                # Remove the position.
                pos.runner_shares = 0.0
                pos.shares = 0.0

        # Clean up fully closed positions.
        self.account.positions = [
            p for p in self.account.positions if p.shares > 0
        ]
        self.update_drawdown(current_price)
        return exits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _close_oldest_position(
        self, shares_sold: float, sell_price: float
    ) -> float:
        """FIFO position close: compute realized PnL."""
        remaining = shares_sold
        total_pnl = 0.0

        for pos in self.account.positions:
            if remaining <= 0:
                break
            if pos.shares <= 0:
                continue

            close_shares = min(remaining, pos.shares)
            entry_cost = close_shares * pos.entry_price
            exit_proceeds = close_shares * sell_price
            pnl = exit_proceeds - entry_cost
            total_pnl += pnl

            pos.shares -= close_shares
            remaining -= close_shares

        # Clean up fully closed positions.
        self.account.positions = [
            p for p in self.account.positions if p.shares > 0
        ]
        self.account.total_directional_pnl += total_pnl
        return total_pnl

    def update_drawdown(self, current_price: float) -> None:
        """Track peak balance and max drawdown using actual mark price."""
        equity = self.account.usdc_balance + (
            self.account.yes_shares * current_price
        )
        if equity > self.account.peak_balance:
            self.account.peak_balance = equity

        if self.account.peak_balance > 0:
            dd = (self.account.peak_balance - equity) / self.account.peak_balance
            self.account.max_drawdown_pct = max(
                self.account.max_drawdown_pct, dd
            )

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized PnL across all open positions."""
        total = 0.0
        for pos in self.account.positions:
            current_value = pos.shares * current_price
            total += current_value - pos.entry_usdc
        self.account.unrealized_pnl = total
        return total

    def get_total_equity(self, current_price: float) -> float:
        """USDC balance + marked-to-market position value."""
        return self.account.usdc_balance + (
            self.account.yes_shares * current_price
        )
