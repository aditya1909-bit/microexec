from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lob.book import LimitOrderBook
from sim.flow import FlowConfig, PoissonOrderFlow


@dataclass
class TwapReport:
    side: str
    target_qty: int
    filled_qty: int
    avg_fill_px: Optional[float]
    arrival_mid: Optional[float]
    shortfall: Optional[float]
    n_child_orders: int
    unfilled_qty: int
    completion_rate: float
    penalty_per_share: float
    penalized_cost: Optional[float]
    shortfall_per_share: Optional[float]
    penalized_cost_per_share: Optional[float]


def _shortfall(side: str, arrival_mid: float, avg_px: float, filled_qty: int) -> float:
    # Signed implementation shortfall vs arrival mid:
    # - BUY (BID): paying above mid => positive cost
    # - SELL (ASK): selling below mid => positive cost
    if side == "BID":
        return (avg_px - arrival_mid) * filled_qty
    else:
        return (arrival_mid - avg_px) * filled_qty


def run_twap(
    *,
    side: str,
    total_qty: int,
    horizon_events: int,
    child_interval: int,
    cfg: FlowConfig,
    seed: int = 0,
    warmup_events: int = 500,
    penalty_per_share: float = 0.0,
) -> tuple[LimitOrderBook, TwapReport]:
    if side not in ("BID", "ASK"):
        raise ValueError(f"side must be 'BID' or 'ASK', got: {side}")
    if total_qty <= 0:
        raise ValueError(f"total_qty must be > 0, got: {total_qty}")
    if horizon_events <= 0:
        raise ValueError(f"horizon_events must be > 0, got: {horizon_events}")
    if child_interval <= 0:
        raise ValueError(f"child_interval must be > 0, got: {child_interval}")
    if penalty_per_share < 0:
        raise ValueError(f"penalty_per_share must be >= 0, got: {penalty_per_share}")

    book = LimitOrderBook()
    flow = PoissonOrderFlow(cfg, seed=seed)

    ts = 0

    # Warmup so the book has quotes and arrival_mid is meaningful.
    for _ in range(warmup_events):
        ts += cfg.dt_us
        flow.step(book, ts)

    arrival_mid = book.mid()

    remaining = total_qty
    filled_qty = 0
    notional = 0  # sum(px * qty) across fills
    n_child = 0

    # Number of scheduled slices across the horizon.
    n_slices = max(1, (horizon_events + child_interval - 1) // child_interval)

    for i in range(horizon_events):
        ts += cfg.dt_us

        # background market event
        flow.step(book, ts)

        # submit TWAP child
        if (i % child_interval) == 0 and remaining > 0:
            slices_left = max(1, n_slices - (i // child_interval))
            child_qty = (remaining + slices_left - 1) // slices_left  # ceil

            fills = book.add_market(side=side, qty=child_qty, ts=ts)
            n_child += 1

            for f in fills:
                filled_qty += f.qty
                notional += f.qty * f.px

            remaining = total_qty - filled_qty
            if remaining <= 0:
                remaining = 0

    avg_px = (notional / filled_qty) if filled_qty > 0 else None

    unfilled_qty = max(0, total_qty - filled_qty)
    completion_rate = filled_qty / total_qty

    if arrival_mid is None or avg_px is None:
        sf = None
        penalized = None
    else:
        sf = _shortfall(side, arrival_mid, avg_px, filled_qty)
        penalized = sf + penalty_per_share * unfilled_qty

    if sf is None or filled_qty == 0:
        sf_per_share = None
    else:
        sf_per_share = sf / filled_qty

    if penalized is None or total_qty == 0:
        penalized_per_share = None
    else:
        penalized_per_share = penalized / total_qty

    report = TwapReport(
        side=side,
        target_qty=total_qty,
        filled_qty=filled_qty,
        avg_fill_px=avg_px,
        arrival_mid=arrival_mid,
        shortfall=sf,
        n_child_orders=n_child,
        unfilled_qty=unfilled_qty,
        completion_rate=completion_rate,
        penalty_per_share=penalty_per_share,
        penalized_cost=penalized,
        shortfall_per_share=sf_per_share,
        penalized_cost_per_share=penalized_per_share,
    )

    return book, report