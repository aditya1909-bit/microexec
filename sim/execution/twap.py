from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lob._cpp import lob_cpp
from lob.book import LimitOrderBook
from sim.flow import FlowConfig, Side


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
    latency_us: int = 0,
    seed: int = 0,
    warmup_events: int = 500,
    penalty_per_share: float = 0.0,
) -> tuple[LimitOrderBook, TwapReport]:
    book, rep = lob_cpp.run_twap(
        side,
        total_qty,
        horizon_events,
        child_interval,
        cfg,
        latency_us,
        seed,
        warmup_events,
        penalty_per_share,
    )

    report = TwapReport(
        side=rep.side,
        target_qty=rep.target_qty,
        filled_qty=rep.filled_qty,
        avg_fill_px=rep.avg_fill_px,
        arrival_mid=rep.arrival_mid,
        shortfall=rep.shortfall,
        n_child_orders=rep.n_child_orders,
        unfilled_qty=rep.unfilled_qty,
        completion_rate=rep.completion_rate,
        penalty_per_share=rep.penalty_per_share,
        penalized_cost=rep.penalized_cost,
        shortfall_per_share=rep.shortfall_per_share,
        penalized_cost_per_share=rep.penalized_cost_per_share,
    )

    return book, report
