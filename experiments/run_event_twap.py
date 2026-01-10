from sim.flow import FlowConfig
from sim.event_sim import ExchangeConfig, TwapAgent, run_event_sim


if __name__ == "__main__":
    cfg = FlowConfig()
    exchange_cfg = ExchangeConfig(
        order_latency_us=2_000,
        md_latency_us=1_000,
        snapshot_interval_us=5_000,
        max_msgs_per_sec=5_000,
        min_resting_us=2_000,
        toxic_prob=0.2,
        toxic_delay_us=5_000,
        toxic_qty_mult=1.0,
    )
    agent = TwapAgent(total_qty=1000, child_interval_events=200)

    book, agent = run_event_sim(
        horizon_events=10_000,
        cfg=cfg,
        exchange_cfg=exchange_cfg,
        agent=agent,
        seed=0,
        use_hawkes=False,
    )

    print("=== event-driven TWAP ===")
    print("filled_qty:", agent.filled_qty)
    print("best_bid:", book.best_bid(), "best_ask:", book.best_ask(), "mid:", book.mid())
