from __future__ import annotations

import argparse
import csv
from itertools import product
import os
from pathlib import Path
from statistics import mean
import math
from typing import Dict, Iterable, List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from sim.event_sim import AvellanedaStoikovAgent, ExchangeConfig, run_event_sim
from sim.flow import FlowConfig, HistoricalFormat


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]

def _parse_bools(text: str) -> List[bool]:
    out: List[bool] = []
    for token in text.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif t in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"invalid boolean value: {token}")
    return out
def _parse_bools(text: str) -> List[bool]:
    out: List[bool] = []
    for token in text.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif t in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"invalid boolean value: {token}")
    return out

def _select_files(historical_dir: str | None, historical_path: str | None, max_files: int) -> List[str]:
    if historical_path:
        return [historical_path]
    if historical_dir:
        data_dir = Path(historical_dir)
        paths = sorted(str(p) for p in data_dir.glob("*.csv"))
        if not paths:
            raise ValueError(f"no csv files found in {data_dir}")
        if max_files > 0:
            return paths[:max_files]
        return paths
    return []


def _worker_run(args: Tuple[dict, str | None, int]) -> Dict[str, float]:
    cfg, path, seed = args
    flow_cfg = FlowConfig()
    exchange_cfg = ExchangeConfig(
        order_latency_us=cfg["order_latency_us"],
        md_latency_us=cfg["md_latency_us"],
        order_latency_jitter_us=cfg["order_latency_jitter_us"],
        md_latency_jitter_us=cfg["md_latency_jitter_us"],
        order_latency_jitter_dist=cfg["order_latency_jitter_dist"],
        md_latency_jitter_dist=cfg["md_latency_jitter_dist"],
        order_latency_logn_mu=cfg["order_latency_logn_mu"],
        order_latency_logn_sigma=cfg["order_latency_logn_sigma"],
        md_latency_logn_mu=cfg["md_latency_logn_mu"],
        md_latency_logn_sigma=cfg["md_latency_logn_sigma"],
        snapshot_interval_us=cfg["snapshot_interval_us"],
        max_msgs_per_sec=cfg["max_msgs_per_sec"],
        min_resting_us=cfg["min_resting_us"],
    )

    agent = AvellanedaStoikovAgent(
        child_interval_events=cfg["child_interval"],
        horizon_events=cfg["horizon_events"],
        risk_aversion=cfg["risk_aversion"],
        kappa=cfg["kappa"],
        base_qty=cfg["base_qty"],
        max_inventory=cfg["max_inventory"],
        sigma_window=cfg["sigma_window"],
        min_spread_ticks=cfg["min_spread_ticks"],
        imbalance_sensitivity=cfg["imbalance_sensitivity"],
        imbalance_level=cfg["imbalance_level"],
        join_best=cfg["join_best"],
        improve_best=cfg["improve_best"],
        improve_ticks=cfg["improve_ticks"],
        cross_prob=cfg["cross_prob"],
        cross_inventory_threshold=cfg["cross_inventory_threshold"],
        cross_ticks=cfg["cross_ticks"],
        seed=seed,
    )

    book, agent = run_event_sim(
        horizon_events=cfg["horizon_events"],
        cfg=flow_cfg,
        exchange_cfg=exchange_cfg,
        agent=agent,
        seed=seed,
        use_historical=cfg["use_historical"],
        historical_path=path if cfg["use_historical"] else None,
        historical_paths=None,
        historical_format=cfg["historical_format"],
        historical_fixed_qty=cfg["fixed_qty"],
        historical_price_scale=cfg["price_scale"],
        warmup_ticks=cfg["warmup_ticks"],
        historical_overlay_p_market=cfg["overlay_p_market"],
        historical_overlay_min_qty=cfg["overlay_min_qty"],
        historical_overlay_max_qty=cfg["overlay_max_qty"],
        historical_overlay_bid_prob=cfg["overlay_bid_prob"],
        overlay_delay_us=cfg["overlay_delay_us"],
        timer_delay_us=cfg["timer_delay_us"],
        debug_first_events=cfg["debug_first_events"],
    )

    mid = book.mid()
    mid_val = float(mid) if mid is not None else 0.0
    mtm = agent.cash + agent.inventory * mid_val
    spread = book.spread() or 0
    mtm_series = list(getattr(agent, "mtm_series", []))
    if mtm_series and mtm_series[-1] != mtm:
        mtm_series.append(mtm)
    delta_sum = 0.0
    delta_sumsq = 0.0
    delta_count = 0
    sharpe_event_run = 0.0
    sharpe_event_adj_run = 0.0
    if len(mtm_series) > 1:
        prev = mtm_series[0]
        for cur in mtm_series[1:]:
            delta = cur - prev
            delta_sum += delta
            delta_sumsq += delta * delta
            delta_count += 1
            prev = cur
        delta_mean = delta_sum / max(1, delta_count)
        delta_var = (delta_sumsq - (delta_sum * delta_sum) / max(1, delta_count)) / max(1, delta_count - 1)
        delta_std = math.sqrt(max(0.0, delta_var))
        inv_abs = abs(float(agent.inventory))
        sharpe_event_run = delta_mean / max(1e-9, delta_std) * math.sqrt(delta_count)
        sharpe_event_adj_run = (
            delta_mean
            / max(1e-9, delta_std + cfg["inv_penalty_lambda"] * inv_abs)
            * math.sqrt(delta_count)
        )

    return {
        **cfg,
        "file": path or "",
        "seed": seed,
        "final_inventory": float(agent.inventory),
        "cash": float(agent.cash),
        "mid": float(mid_val),
        "mtm": float(mtm),
        "spread": float(spread),
        "n_fills": float(agent.n_fills),
        "filled_qty": float(agent.filled_qty),
        "delta_sum": float(delta_sum),
        "delta_sumsq": float(delta_sumsq),
        "delta_count": float(delta_count),
        "sharpe_event_run": float(sharpe_event_run),
        "sharpe_event_adj_run": float(sharpe_event_adj_run),
    }


def _grid(cfg: dict) -> Iterable[dict]:
    keys = [
        "risk_aversion",
        "kappa",
        "base_qty",
        "overlay_p_market",
        "join_best",
        "improve_best",
        "imbalance_sensitivity",
    ]
    values = [cfg[k] for k in keys]
    for (
        ra,
        kappa,
        base_qty,
        overlay_p_market,
        join_best,
        improve_best,
        imbalance_sensitivity,
    ) in product(*values):
        out = dict(cfg)
        out["risk_aversion"] = ra
        out["kappa"] = kappa
        out["base_qty"] = base_qty
        out["overlay_p_market"] = overlay_p_market
        out["join_best"] = join_best
        out["improve_best"] = improve_best
        out["imbalance_sensitivity"] = imbalance_sensitivity
        yield out


def _summarize(rows: Sequence[Dict[str, float]], inv_penalty_lambda: float) -> Dict[str, float]:
    if not rows:
        return {
            "mtm_mean": 0.0,
            "mtm_std": 0.0,
            "mtm_abs_mean": 0.0,
            "inv_mean": 0.0,
            "inv_abs_mean": 0.0,
            "spread_mean": 0.0,
            "sharpe_event_mean": 0.0,
            "sharpe_event_adj_mean": 0.0,
            "sharpe_event_pooled": 0.0,
            "sharpe_event_adj_pooled": 0.0,
            "fills_mean": 0.0,
            "filled_qty_mean": 0.0,
            "delta_count": 0.0,
        }
    mtm_vals = [r["mtm"] for r in rows]
    mtm_mean = mean(mtm_vals)
    mtm_std = (sum((x - mtm_mean) ** 2 for x in mtm_vals) / max(1, len(mtm_vals) - 1)) ** 0.5
    inv_abs_mean = mean(abs(r["final_inventory"]) for r in rows)
    delta_sum_total = sum(r.get("delta_sum", 0.0) for r in rows)
    delta_sumsq_total = sum(r.get("delta_sumsq", 0.0) for r in rows)
    delta_count_total = sum(int(r.get("delta_count", 0)) for r in rows)
    if delta_count_total > 1:
        delta_mean_total = delta_sum_total / delta_count_total
        delta_var_total = (
            delta_sumsq_total - (delta_sum_total * delta_sum_total) / delta_count_total
        ) / (delta_count_total - 1)
        delta_std_total = math.sqrt(max(0.0, delta_var_total))
    else:
        delta_mean_total = 0.0
        delta_std_total = 0.0
    sharpe_event_pooled = delta_mean_total / max(1e-9, delta_std_total) * math.sqrt(
        max(1, delta_count_total)
    )
    sharpe_event_adj_pooled = delta_mean_total / max(
        1e-9, delta_std_total + inv_penalty_lambda * inv_abs_mean
    ) * math.sqrt(max(1, delta_count_total))
    sharpe_event_mean = mean(r.get("sharpe_event_run", 0.0) for r in rows)
    sharpe_event_adj_mean = mean(r.get("sharpe_event_adj_run", 0.0) for r in rows)
    delta_count = mean(r.get("delta_count", 0.0) for r in rows)
    return {
        "mtm_mean": mtm_mean,
        "mtm_std": mtm_std,
        "mtm_abs_mean": mean(abs(r["mtm"]) for r in rows),
        "inv_mean": mean(r["final_inventory"] for r in rows),
        "inv_abs_mean": inv_abs_mean,
        "spread_mean": mean(r["spread"] for r in rows),
        "sharpe_event_mean": sharpe_event_mean,
        "sharpe_event_adj_mean": sharpe_event_adj_mean,
        "sharpe_event_pooled": sharpe_event_pooled,
        "sharpe_event_adj_pooled": sharpe_event_adj_pooled,
        "fills_mean": mean(r["n_fills"] for r in rows),
        "filled_qty_mean": mean(r["filled_qty"] for r in rows),
        "delta_count": float(delta_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-events", type=int, default=50_000)
    parser.add_argument("--child-interval", type=int, default=50)
    parser.add_argument("--risk-aversion", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--kappa", type=str, default="0.5,0.2,0.1,0.05,0.01")
    parser.add_argument("--base-qty", type=str, default="1,2")
    parser.add_argument("--max-inventory", type=int, default=50)
    parser.add_argument("--sigma-window", type=int, default=200)
    parser.add_argument("--min-spread-ticks", type=int, default=1)
    parser.add_argument("--imbalance-sensitivity", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--imbalance-level", type=int, default=1)
    parser.add_argument("--join-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--join-best-values", type=str, default="true,false")
    parser.add_argument("--improve-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--improve-best-values", type=str, default="true")
    parser.add_argument("--improve-ticks", type=int, default=1)
    parser.add_argument("--cross-prob", type=float, default=0.05)
    parser.add_argument("--cross-inventory-threshold", type=int, default=1)
    parser.add_argument("--cross-ticks", type=int, default=1)

    parser.add_argument("--historical-path", type=str, default=None)
    parser.add_argument("--historical-dir", type=str, default=None)
    parser.add_argument("--historical-format", type=str, default="truefx")
    parser.add_argument("--fixed-qty", type=int, default=1_000_000)
    parser.add_argument("--price-scale", type=int, default=100_000)
    parser.add_argument("--warmup-ticks", type=int, default=500)

    parser.add_argument("--overlay-p-market", type=str, default="0.01,0.02,0.05")
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

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=-1)
    parser.add_argument("--inv-penalty-lambda", type=float, default=0.1)
    args = parser.parse_args()

    fmt_map = {
        "truefx": HistoricalFormat.TRUEFX,
        "csv": HistoricalFormat.CSV,
        "binary": HistoricalFormat.BINARY,
    }
    fmt_key = args.historical_format.strip().lower()
    if fmt_key not in fmt_map:
        raise ValueError(f"unknown historical_format: {args.historical_format}")

    files = _select_files(args.historical_dir, args.historical_path, args.max_files)
    if files:
        raise ValueError("historical inputs are disabled for this grid search; use simulated flow only")
    use_historical = False

    cfg = {
        "horizon_events": args.horizon_events,
        "child_interval": args.child_interval,
        "risk_aversion": _parse_floats(args.risk_aversion),
        "kappa": _parse_floats(args.kappa),
        "base_qty": _parse_ints(args.base_qty),
        "max_inventory": args.max_inventory,
        "sigma_window": args.sigma_window,
        "min_spread_ticks": args.min_spread_ticks,
        "imbalance_sensitivity": _parse_floats(args.imbalance_sensitivity),
        "imbalance_level": args.imbalance_level,
        "join_best": _parse_bools(args.join_best_values),
        "improve_best": _parse_bools(args.improve_best_values),
        "improve_ticks": args.improve_ticks,
        "cross_prob": args.cross_prob,
        "cross_inventory_threshold": args.cross_inventory_threshold,
        "cross_ticks": args.cross_ticks,
        "use_historical": use_historical,
        "historical_format": fmt_map[fmt_key],
        "fixed_qty": args.fixed_qty,
        "price_scale": args.price_scale,
        "warmup_ticks": args.warmup_ticks,
        "overlay_p_market": _parse_floats(args.overlay_p_market),
        "overlay_min_qty": args.overlay_min_qty,
        "overlay_max_qty": args.overlay_max_qty,
        "overlay_bid_prob": args.overlay_bid_prob,
        "overlay_delay_us": args.overlay_delay_us,
        "timer_delay_us": args.timer_delay_us,
        "debug_first_events": args.debug_first_events,
        "inv_penalty_lambda": args.inv_penalty_lambda,
        "order_latency_us": args.order_latency_us,
        "md_latency_us": args.md_latency_us,
        "order_latency_jitter_us": args.order_latency_jitter_us,
        "md_latency_jitter_us": args.md_latency_jitter_us,
        "order_latency_jitter_dist": args.order_latency_jitter_dist,
        "md_latency_jitter_dist": args.md_latency_jitter_dist,
        "order_latency_logn_mu": args.order_latency_logn_mu,
        "order_latency_logn_sigma": args.order_latency_logn_sigma,
        "md_latency_logn_mu": args.md_latency_logn_mu,
        "md_latency_logn_sigma": args.md_latency_logn_sigma,
        "snapshot_interval_us": args.snapshot_interval_us,
        "max_msgs_per_sec": args.max_msgs_per_sec,
        "min_resting_us": args.min_resting_us,
    }

    seeds = _parse_ints(args.seeds)
    if not seeds:
        raise ValueError("seeds must be non-empty")

    tasks: List[Tuple[dict, str | None, int]] = []
    for grid_cfg in _grid(cfg):
        for seed in seeds:
            tasks.append((grid_cfg, None, seed))

    worker_count = args.max_workers
    if worker_count <= 0:
        worker_count = max(1, min(os.cpu_count() or 2, len(tasks)))

    rows: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_worker_run, t) for t in tasks]
        for future in as_completed(futures):
            rows.append(future.result())

    grouped: Dict[Tuple[float, float, int, float, bool, bool, float, int], List[Dict[str, float]]] = {}
    for row in rows:
        key = (
            row["risk_aversion"],
            row["kappa"],
            row["base_qty"],
            row["overlay_p_market"],
            bool(row["join_best"]),
            bool(row["improve_best"]),
            row["imbalance_sensitivity"],
            int(row["imbalance_level"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, float]] = []
    for key, bucket in grouped.items():
        ra, kappa, base_qty, overlay_p_market, join_best, improve_best, imbalance_sensitivity, imbalance_level = key
        summary = _summarize(bucket, cfg["inv_penalty_lambda"])
        summary_rows.append(
            {
                "risk_aversion": ra,
                "kappa": kappa,
                "base_qty": base_qty,
                "overlay_p_market": overlay_p_market,
                "join_best": join_best,
                "improve_best": improve_best,
                "imbalance_sensitivity": imbalance_sensitivity,
                "imbalance_level": imbalance_level,
                "inv_penalty_lambda": cfg["inv_penalty_lambda"],
                **summary,
            }
        )

    summary_rows.sort(key=lambda r: r["sharpe_event_adj_mean"], reverse=True)

    out_dir = Path("experiments/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mm_as_grid.csv"
    with open(out_path, "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"saved grid summary to {out_path}")
    print("top 5 configs:")
    for row in summary_rows[:5]:
        print(
            "ra={risk_aversion:.3f} kappa={kappa:.2f} base_qty={base_qty} "
            "overlay_p={overlay_p_market:.3f} join_best={join_best} improve_best={improve_best} "
            "imb_sense={imbalance_sensitivity:.2f} sharpe_event_mean={sharpe_event_mean:.3f} "
            "sharpe_event_adj_mean={sharpe_event_adj_mean:.3f} "
            "sharpe_event_pooled={sharpe_event_pooled:.3f} "
            "sharpe_event_adj_pooled={sharpe_event_adj_pooled:.3f} mtm_mean={mtm_mean:.2f} "
            "inv_abs={inv_abs_mean:.2f} "
            "fills_mean={fills_mean:.1f} filled_qty_mean={filled_qty_mean:.1f}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
