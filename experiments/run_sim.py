from sim.engine import run_sim
from sim.flow import FlowConfig


if __name__ == "__main__":
    # Use FlowConfig defaults from sim/flow.py so tuning changes take effect.
    config = FlowConfig()

    book, stats = run_sim(n_events = 50000, config = config, seed = 0)

    print("=== microexec sim v0 ===")
    print(f"events: {stats.n_events}")
    print(f"limit: {stats.n_limit} | market: {stats.n_market} | cancel: {stats.n_cancel}")
    print(f"total traded qty: {stats.traded_qty}")
    print(f"avg spread (when defined): {stats.avg_spread():.3f}")
    print(f"best bid: {book.best_bid()} | best ask: {book.best_ask()} | mid: {book.mid()}")

    print("top depth (BID):", book.depth("BID", levels=5))
    print("top depth (ASK):", book.depth("ASK", levels=5))
    print("imbalance (1):", book.imbalance(levels=1))