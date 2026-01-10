from sim.engine import run_sim
from sim.flow import FlowConfig, Side
from experiments.progress import progress_bar


if __name__ == "__main__":
    # Use FlowConfig defaults from sim/flow.py so tuning changes take effect.
    config = FlowConfig()

    progress_bar(0, 1, prefix="run_sim")
    book, stats = run_sim(n_events = 50000, config = config, seed = 0)
    progress_bar(1, 1, prefix="run_sim")

    print("=== microexec sim v0 ===")
    print(f"events: {stats.n_events}")
    print(f"limit: {stats.n_limit} | market: {stats.n_market} | cancel: {stats.n_cancel}")
    print(f"total traded qty: {stats.traded_qty}")
    print(f"avg spread (when defined): {stats.avg_spread():.3f}")
    print(f"best bid: {book.best_bid()} | best ask: {book.best_ask()} | mid: {book.mid()}")

    print("top depth (BID):", book.depth(Side.BID, levels=5))
    print("top depth (ASK):", book.depth(Side.ASK, levels=5))
    print("imbalance (1):", book.imbalance(levels=1))
