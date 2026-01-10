from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lob._cpp import lob_cpp
from lob.book import LimitOrderBook
from sim.flow import FlowConfig


@dataclass
class ImpactModel:
    temporary_impact: float = 0.0
    permanent_impact: float = 0.0
    volatility: float = 0.0
    risk_aversion: float = 0.0


@dataclass
class AlmgrenChrissReport:
    side: str
    target_qty: int
    filled_qty: int
    avg_fill_px: Optional[float]
    arrival_mid: Optional[float]
    shortfall: Optional[float]
    n_child_orders: int
    unfilled_qty: int
    completion_rate: float
    risk_penalty: Optional[float]
    objective: Optional[float]
    shortfall_uncapped: Optional[float]
    objective_uncapped: Optional[float]
    cap_abs: int
    cap_frac: float
    cap_used: int
    capped: bool
    child_qtys: list[int]
    impact: ImpactModel


def run_almgren_chriss(
    *,
    side: str,
    total_qty: int,
    horizon_events: int,
    child_interval: int,
    cfg: FlowConfig,
    impact: ImpactModel,
    cap_abs: int = 0,
    cap_frac: float = 0.0,
    latency_us: int = 0,
    seed: int = 0,
    warmup_events: int = 500,
) -> tuple[LimitOrderBook, AlmgrenChrissReport]:
    impact_cpp = lob_cpp.ImpactModel()
    impact_cpp.temporary_impact = impact.temporary_impact
    impact_cpp.permanent_impact = impact.permanent_impact
    impact_cpp.volatility = impact.volatility
    impact_cpp.risk_aversion = impact.risk_aversion

    book, rep = lob_cpp.run_almgren_chriss(
        side,
        total_qty,
        horizon_events,
        child_interval,
        cfg,
        impact_cpp,
        cap_abs,
        cap_frac,
        latency_us,
        seed,
        warmup_events,
    )

    report = AlmgrenChrissReport(
        side=rep.side,
        target_qty=rep.target_qty,
        filled_qty=rep.filled_qty,
        avg_fill_px=rep.avg_fill_px,
        arrival_mid=rep.arrival_mid,
        shortfall=rep.shortfall,
        n_child_orders=rep.n_child_orders,
        unfilled_qty=rep.unfilled_qty,
        completion_rate=rep.completion_rate,
        risk_penalty=rep.risk_penalty,
        objective=rep.objective,
        shortfall_uncapped=rep.shortfall_uncapped,
        objective_uncapped=rep.objective_uncapped,
        cap_abs=rep.cap_abs,
        cap_frac=rep.cap_frac,
        cap_used=rep.cap_used,
        capped=rep.capped,
        child_qtys=list(rep.child_qtys),
        impact=impact,
    )

    return book, report
