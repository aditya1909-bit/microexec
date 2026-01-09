from lob._cpp import lob_cpp
from .flow import FlowConfig

LimitOrderBook = lob_cpp.LimitOrderBook
SimStats = lob_cpp.SimStats
run_sim_cpp = lob_cpp.run_sim

__all__ = ["SimStats", "run_sim", "FlowConfig", "LimitOrderBook"]


def run_sim(n_events: int, config: FlowConfig, seed: int = 0) -> tuple[LimitOrderBook, SimStats]:
    return run_sim_cpp(n_events, config, seed)
