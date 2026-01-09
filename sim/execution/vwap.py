from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lob._cpp import lob_cpp
from lob.book import LimitOrderBook
from sim.flow import FlowConfig

@dataclass
class VwapReport:
    side: str
    target_qty: int
    filled_qty: int
    avg_fill_px: Optional[float]
    arrival_mid: Optional[float]
    shortfall: Optional[float]
    n_child_orders: int
    unfilled_qty: int
    completion_rate: float
    penalty_per_share: Optional[float]
    penalized_cost: Optional[float]
    shortfall_per_share: Optional[float]
    penalized_cost_per_share: Optional[float]
    
    n_buckets: int
    bucket_interval: int
    forecast_total_mkt_vol: int
    
def run_vwap(
    *,
    side: str,
    total_qty: int,
    horizon_events: int,
    bucket_interval: int,
    cfg: FlowConfig,
    seed: int = 0,
    warmup_events: int = 500,
    penalty_per_share: float = 0.0,
) -> tuple[LimitOrderBook, VwapReport]:
    book, rep = lob_cpp.run_vwap(
        side,
        total_qty,
        horizon_events,
        bucket_interval,
        cfg,
        seed,
        warmup_events,
        penalty_per_share,
    )

    report = VwapReport(
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
        n_buckets=rep.n_buckets,
        bucket_interval=rep.bucket_interval,
        forecast_total_mkt_vol=rep.forecast_total_mkt_vol,
    )

    return book, report
