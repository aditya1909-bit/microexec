from __future__ import annotations

from dataclasses import dataclass

from lob.book import LimitOrderBook
from .flow import PoissonOrderFlow, FlowConfig

@dataclass
class SimStats:
    n_events: int = 0
    n_limit: int = 0
    n_market: int = 0
    n_cancel: int = 0
    traded_qty: int = 0
    
    spread_samples: int = 0
    spread_sum: int = 0
    
    def record_spread(self, book: LimitOrderBook):
        s = book.spread()
        if s is not None:
            self.spread_sum += s
            self.spread_samples += 1
    
    def avg_spread(self) -> float:
        if self.spread_samples == 0:
            return float("nan")
        return self.spread_sum / self.spread_samples
    
def run_sim(n_events: int, config: FlowConfig, seed: int = 0) -> tuple[LimitOrderBook, SimStats]:
    book = LimitOrderBook()
    flow = PoissonOrderFlow(config, seed)
    stats = SimStats()
    
    ts = 0
    
    for _ in range(n_events):
        ts += config.dt_us
        kind, traded = flow.step(book, ts)
        
        stats.n_events += 1
        stats.traded_qty += traded
        
        if kind == "LIMIT":
            stats.n_limit += 1
        elif kind == "MARKET":
            stats.n_market += 1
        elif kind == "CANCEL":
            stats.n_cancel += 1
        
        stats.record_spread(book)
    
    return book, stats