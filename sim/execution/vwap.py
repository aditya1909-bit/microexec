from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lob.book import LimitOrderBook
from sim.flow import FlowConfig, PoissonOrderFlow

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
    penalty_per_share: float
    penalized_cost: Optional[float]
    shortfall_per_share: Optional[float]
    penalized_cost_per_share: Optional[float]
    
    n_buckets: int
    bucket_interval: int
    forecast_total_mkt_vol: int
    
def _shortfall(side: str, arrival_mid: float, avg_px: float, filled_qty: int) -> float:
    if side == "BID":
        return (avg_px - arrival_mid) * filled_qty
    else:
        return (arrival_mid - avg_px) * filled_qty
    
def _computer_bucket_targets(total_qty: int, vols: list[int]) -> list[int]:
    n = len(vols)
    
    if n == 0:
        return []
    if total_qty <= 0:
        return [0] * n
    
    total_vol = sum(vols)
    
    if total_vol <= 0:
        base = total_qty // n
        rem = total_qty - base * n
        out = [base] * n
        for i in range(rem):
            out[i] += 1
        return out
    
    raw = [total_qty * (v / total_vol) for v in vols]
    floors = [int(x) for x in raw]
    rem = total_qty - sum(floors)
    
    fracs = [(raw[i] - floors[i], i) for i in range (n)]
    fracs.sort(reverse = True)
    
    out = floors[:]
    for k in range(rem):
        out[fracs[k][1]] += 1
        
    s = sum(out)
    if s != total_qty:
        out[-1] += total_qty - s
    
    return out

def _forecast_market_volume(
    *,
    horizon_events: int,
    bucket_interval: int, 
    cfg: FlowConfig,
    seed: int,
    warmup_events: int
) -> list[int]:
    book = LimitOrderBook()
    flow = PoissonOrderFlow(cfg, seed = seed)
    
    ts = 0
    for _ in range(warmup_events):
        ts += cfg.dt_us
        flow.step(book, ts)
    
    n_buckets = max(1, (horizon_events + bucket_interval - 1) // bucket_interval)
    vols = [0] * n_buckets
    
    for i in range(horizon_events):
        ts += cfg.dt_us
        kind, traded = flow.step(book, ts)
        if kind == "MARKET":
            vols[i // bucket_interval] += int(traded)
    
    return vols

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
    
    if side not in ("BID", "ASK"):
        raise ValueError(f"Side must be 'BID' or 'ASK', got: {side}")
    if total_qty <= 0:
        raise ValueError(f"Total Qty must be > 0, got: {total_qty}")
    if horizon_events <= 0:
        raise ValueError(f"Horizon Events must be > 0, got: {horizon_events}")
    if bucket_interval <= 0:
        raise ValueError(f"Bucket Interval must be > 0, got: {bucket_interval}")
    if penalty_per_share < 0:
        raise ValueError(f"Penalty Per Share must be >= 0, got: {penalty_per_share}")
    
    vols = _forecast_market_volume(
        horizon_events = horizon_events,
        bucket_interval = bucket_interval,
        cfg = cfg,
        seed = seed,
        warmup_events = warmup_events,
    )
    
    n_buckets = len(vols)
    forecast_total_mkt_vol = sum(vols)
    
    bucket_targets = _computer_bucket_targets(total_qty, vols)
    
    book = LimitOrderBook()
    flow = PoissonOrderFlow(cfg, seed = seed)
    
    ts = 0
    
    for _ in range(warmup_events):
        ts += cfg.dt_us
        flow.step(book, ts)
        
    arrival_mid = book.mid()
    
    filled_qty = 0
    notional = 0
    n_child = 0
    
    remaining = total_qty
    
    for i in range(horizon_events):
        ts += cfg.dt_us
        
        flow.step(book, ts)
        
        if (i % bucket_interval) == 0 and remaining > 0:
            b = i // bucket_interval
            
            if b < n_buckets:
                child_qty = min(remaining, int(bucket_targets[b]))
            else:
                child_qty = remaining
            
            if child_qty > 0:
                fills = book.add_market(side = side, qty = child_qty, ts = ts)
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
        penalty_per_share = None
    else:
        penalty_per_share = penalized / total_qty
    
    report = VwapReport(
        side = side,
        target_qty = total_qty,
        filled_qty = filled_qty,
        avg_fill_px = avg_px,
        arrival_mid = arrival_mid,
        shortfall = sf,
        n_child_orders = n_child,
        unfilled_qty = unfilled_qty,
        completion_rate = completion_rate,
        penalty_per_share = penalty_per_share,
        penalized_cost = penalized,
        shortfall_per_share = sf_per_share,
        penalized_cost_per_share = penalty_per_share,
        n_buckets = n_buckets,
        bucket_interval = bucket_interval,
        forecast_total_mkt_vol = forecast_total_mkt_vol,
    )
    
    return book, report