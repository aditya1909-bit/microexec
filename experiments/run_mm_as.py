from __future__ import annotations

import argparse
from pathlib import Path

from sim.event_sim import ExchangeConfig, AvellanedaStoikovAgent, run_event_sim
from sim.flow import FlowConfig, HistoricalFormat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-events", type=int, default=10_000)
    parser.add_argument("--child-interval", type=int, default=50)
    parser.add_argument("--risk-aversion", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=1.5)
    parser.add_argument("--base-qty", type=int, default=1)
    parser.add_argument("--max-inventory", type=int, default=50)
    parser.add_argument("--sigma-window", type=int, default=200)
    parser.add_argument("--min-spread-ticks", type=int, default=1)
    parser.add_argument("--imbalance-sensitivity", type=float, default=0.0)
    parser.add_argument("--imbalance-level", type=int, default=1)
    parser.add_argument("--join-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--improve-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--improve-ticks", type=int, default=1)
    parser.add_argument("--cross-prob", type=float, default=0.05)
    parser.add_argument("--cross-inventory-threshold", type=int, default=1)
    parser.add_argument("--cross-ticks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--historical-path", type=str, default=None)
    parser.add_argument("--historical-dir", type=str, default=None)
    parser.add_argument("--historical-format", type=str, default="truefx")
    parser.add_argument("--fixed-qty", type=int, default=1_000_000)
    parser.add_argument("--price-scale", type=int, default=100_000)
    parser.add_argument("--warmup-ticks", type=int, default=500)
    parser.add_argument("--overlay-p-market", type=float, default=0.05)
    parser.add_argument("--overlay-min-qty", type=int, default=1)
    parser.add_argument("--overlay-max-qty", type=int, default=5)
    parser.add_argument("--overlay-bid-prob", type=float, default=0.5)
    parser.add_argument("--overlay-delay-us", type=int, default=1)
    parser.add_argument("--timer-delay-us", type=int, default=1)
    parser.add_argument("--debug-first-events", type=int, default=0)
    parser.add_argument("--order-latency-us", type=int, default=0)
    parser.add_argument("--order-latency-jitter-us", type=int, default=0)
    parser.add_argument("--md-latency-us", type=int, default=0)
    parser.add_argument("--md-latency-jitter-us", type=int, default=0)
    parser.add_argument("--order-latency-jitter-dist", type=str, default="uniform")
    parser.add_argument("--md-latency-jitter-dist", type=str, default="uniform")
    parser.add_argument("--order-latency-logn-mu", type=float, default=0.0)
    parser.add_argument("--order-latency-logn-sigma", type=float, default=0.0)
    parser.add_argument("--md-latency-logn-mu", type=float, default=0.0)
    parser.add_argument("--md-latency-logn-sigma", type=float, default=0.0)
    parser.add_argument("--snapshot-interval-us", type=int, default=0)
    parser.add_argument("--max-msgs-per-sec", type=int, default=0)
    parser.add_argument("--min-resting-us", type=int, default=0)
    args = parser.parse_args()

    fmt_map = {
        "truefx": HistoricalFormat.TRUEFX,
        "csv": HistoricalFormat.CSV,
        "binary": HistoricalFormat.BINARY,
    }
    fmt_key = args.historical_format.strip().lower()
    if fmt_key not in fmt_map:
        raise ValueError(f"unknown historical_format: {args.historical_format}")

    historical_paths = None
    if args.historical_dir:
        data_dir = Path(args.historical_dir)
        historical_paths = sorted(str(p) for p in data_dir.glob("*.csv"))
        if not historical_paths:
            raise ValueError(f"no csv files found in {data_dir}")

    use_historical = args.historical_path is not None or historical_paths is not None

    flow_cfg = FlowConfig()
    exchange_cfg = ExchangeConfig(
        order_latency_us=args.order_latency_us,
        md_latency_us=args.md_latency_us,
        order_latency_jitter_us=args.order_latency_jitter_us,
        md_latency_jitter_us=args.md_latency_jitter_us,
        order_latency_jitter_dist=args.order_latency_jitter_dist,
        md_latency_jitter_dist=args.md_latency_jitter_dist,
        order_latency_logn_mu=args.order_latency_logn_mu,
        order_latency_logn_sigma=args.order_latency_logn_sigma,
        md_latency_logn_mu=args.md_latency_logn_mu,
        md_latency_logn_sigma=args.md_latency_logn_sigma,
        snapshot_interval_us=args.snapshot_interval_us,
        max_msgs_per_sec=args.max_msgs_per_sec,
        min_resting_us=args.min_resting_us,
    )

    agent = AvellanedaStoikovAgent(
        child_interval_events=args.child_interval,
        horizon_events=args.horizon_events,
        risk_aversion=args.risk_aversion,
        kappa=args.kappa,
        base_qty=args.base_qty,
        max_inventory=args.max_inventory,
        sigma_window=args.sigma_window,
        min_spread_ticks=args.min_spread_ticks,
        imbalance_sensitivity=args.imbalance_sensitivity,
        imbalance_level=args.imbalance_level,
        join_best=args.join_best,
        improve_best=args.improve_best,
        improve_ticks=args.improve_ticks,
        cross_prob=args.cross_prob,
        cross_inventory_threshold=args.cross_inventory_threshold,
        cross_ticks=args.cross_ticks,
        seed=args.seed,
    )

    book, agent = run_event_sim(
        horizon_events=args.horizon_events,
        cfg=flow_cfg,
        exchange_cfg=exchange_cfg,
        agent=agent,
        seed=args.seed,
        use_historical=use_historical,
        historical_path=args.historical_path,
        historical_paths=historical_paths,
        historical_format=fmt_map[fmt_key],
        historical_fixed_qty=args.fixed_qty,
        historical_price_scale=args.price_scale,
        warmup_ticks=args.warmup_ticks,
        historical_overlay_p_market=args.overlay_p_market,
        historical_overlay_min_qty=args.overlay_min_qty,
        historical_overlay_max_qty=args.overlay_max_qty,
        historical_overlay_bid_prob=args.overlay_bid_prob,
        overlay_delay_us=args.overlay_delay_us,
        timer_delay_us=args.timer_delay_us,
        debug_first_events=args.debug_first_events,
    )

    mid = book.mid()
    mid_val = float(mid) if mid is not None else 0.0
    mtm = agent.cash + agent.inventory * mid_val

    print("=== Avellaneda-Stoikov MM ===")
    print(f"use_historical={use_historical} horizon_events={args.horizon_events}")
    print(f"final_inventory={agent.inventory} cash={agent.cash:.2f} mid={mid_val:.2f} mtm={mtm:.2f}")
    print(f"fills={agent.n_fills} filled_qty={agent.filled_qty} orders={agent.n_orders}")
    print(f"best_bid={book.best_bid()} best_ask={book.best_ask()} spread={book.spread()}")


if __name__ == "__main__":
    main()
