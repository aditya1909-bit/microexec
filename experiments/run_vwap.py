from __future__ import annotations

from dataclasses import asdict
import os
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

from joblib import Parallel, delayed
from experiments.progress import progress_bar

from sim.flow import FlowConfig, Side
from sim.execution.twap import run_twap
from sim.execution.vwap import run_vwap


def _mean_std(xs):
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), pstdev(xs)


def _ci95(std: float | None, n: int) -> float:
    if std is None or n <= 0:
        return 0.0
    return 1.96 * float(std) / sqrt(n)


def _impact_per_share(shortfall_per_share: float | None) -> float | None:
    if shortfall_per_share is None:
        return None
    # Baseline: crossing spread is ~1 tick for BID market orders.
    return max(0.0, float(shortfall_per_share) - 1.0)


def _run_one(
    algo: str,
    seed: int,
    total_qty: int,
    interval: int,
    penalty_per_share: float,
    horizon_events: int,
    warmup_events: int,
) -> dict:
    cfg = FlowConfig()

    if algo == "TWAP":
        book, rep = run_twap(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=interval,
            cfg=cfg,
            seed=seed,
            warmup_events=warmup_events,
            penalty_per_share=penalty_per_share,
        )
    elif algo == "VWAP":
        book, rep = run_vwap(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            bucket_interval=interval,
            cfg=cfg,
            seed=seed,
            warmup_events=warmup_events,
            penalty_per_share=penalty_per_share,
        )
    else:
        raise ValueError(f"unknown algo: {algo}")

    d = asdict(rep)
    d["algo"] = algo
    d["seed"] = seed
    d["total_qty"] = total_qty
    d["interval"] = interval
    d["penalty_per_share"] = penalty_per_share
    d["impact_per_share"] = _impact_per_share(d.get("shortfall_per_share"))

    bid_touch = book.depth(Side.BID, levels=1)
    ask_touch = book.depth(Side.ASK, levels=1)
    d["end_bid_touch"] = bid_touch[0][1] if bid_touch else 0
    d["end_ask_touch"] = ask_touch[0][1] if ask_touch else 0
    d["end_imbalance_1"] = book.imbalance(levels=1)

    return d


if __name__ == "__main__":
    # Sweep parameters
    algos = ["TWAP", "VWAP"]

    total_qty_grid = [250, 500, 1000, 2000]
    interval_grid = [50, 100, 200, 500, 1000]  # child_interval for TWAP, bucket_interval for VWAP
    penalty_grid = [0.0, 1.0, 2.0, 5.0]

    seed_grid = list(range(20))

    horizon_events = 10_000
    warmup_events = 500

    # Build task list
    tasks = []
    for algo in algos:
        for seed in seed_grid:
            for total_qty in total_qty_grid:
                for interval in interval_grid:
                    for penalty_per_share in penalty_grid:
                        tasks.append((algo, seed, total_qty, interval, penalty_per_share))

    cpu = os.cpu_count() or 1
    chunk_size = max(1, len(tasks) // (cpu * 4))
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    rows = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        rows_chunk = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(algo, seed, total_qty, interval, penalty_per_share, horizon_events, warmup_events)
            for algo, seed, total_qty, interval, penalty_per_share in chunk
        )
        rows.extend(rows_chunk)
        progress_bar(idx, total_chunks, prefix="vwap")

    # Aggregate by (algo, total_qty, interval, penalty_per_share)
    grouped: dict[tuple[str, int, int, float], list[dict]] = {}
    for r in rows:
        key = (r["algo"], r["total_qty"], r["interval"], r["penalty_per_share"])
        grouped.setdefault(key, []).append(r)

    agg = []
    for (algo, total_qty, interval, penalty_per_share), rs in grouped.items():
        n = len(rs)

        pcs = [x.get("penalized_cost_per_share") for x in rs if x.get("penalized_cost_per_share") is not None]
        sps = [x.get("shortfall_per_share") for x in rs if x.get("shortfall_per_share") is not None]
        ips = [x.get("impact_per_share") for x in rs if x.get("impact_per_share") is not None]
        crs = [x.get("completion_rate") for x in rs if x.get("completion_rate") is not None]

        pcs_m, pcs_s = _mean_std(pcs)
        sps_m, sps_s = _mean_std(sps)
        ips_m, ips_s = _mean_std(ips)
        cr_m, cr_s = _mean_std(crs)

        agg.append(
            {
                "algo": algo,
                "total_qty": total_qty,
                "interval": interval,
                "penalty_per_share": penalty_per_share,
                "n_seeds": n,
                "penalized_cost_per_share_mean": pcs_m,
                "penalized_cost_per_share_std": pcs_s,
                "shortfall_per_share_mean": sps_m,
                "shortfall_per_share_std": sps_s,
                "impact_per_share_mean": ips_m,
                "impact_per_share_std": ips_s,
                "completion_rate_mean": cr_m,
                "completion_rate_std": cr_s,
            }
        )

    penalty_to_plot = 2.0

    def pick(algo: str, total_qty: int, interval: int):
        return next(
            (
                a
                for a in agg
                if a["algo"] == algo
                and a["total_qty"] == total_qty
                and a["interval"] == interval
                and a["penalty_per_share"] == penalty_to_plot
            ),
            None,
        )

    print("=== TWAP vs VWAP (mean±std over seeds) @ penalty_per_share=2.0 ===")
    print("algo | qty | interval | penalized_mean±std | shortfall_mean±std | impact_mean±std | completion_mean±std")
    for total_qty in total_qty_grid:
        for interval in interval_grid:
            for algo in algos:
                a = pick(algo, total_qty, interval)
                if a is None:
                    continue
                pm = a.get("penalized_cost_per_share_mean")
                ps = a.get("penalized_cost_per_share_std")
                sm = a.get("shortfall_per_share_mean")
                ss = a.get("shortfall_per_share_std")
                im = a.get("impact_per_share_mean")
                istd = a.get("impact_per_share_std")
                cm = a.get("completion_rate_mean")
                cs = a.get("completion_rate_std")

                def fmt(m, s):
                    if m is None:
                        return "None"
                    if s is None:
                        return f"{m:.4f}"
                    return f"{m:.4f}±{s:.4f}"

                print(
                    f"{algo:>4} | {total_qty:>4} | {interval:>8} | "
                    f"{fmt(pm, ps):>18} | "
                    f"{fmt(sm, ss):>18} | "
                    f"{fmt(im, istd):>14} | "
                    f"{fmt(cm, cs):>16}"
                )

    # Plots (comparison)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available; skipping plots. ({e})")
        raise SystemExit(0)

    def series(metric_mean: str, metric_std: str, algo: str, total_qty: int):
        xs, ys, es = [], [], []
        for interval in interval_grid:
            a = pick(algo, total_qty, interval)
            if a is None:
                continue
            y = a.get(metric_mean)
            s = a.get(metric_std)
            if y is None:
                continue
            xs.append(interval)
            ys.append(y)
            es.append(_ci95(s, a["n_seeds"]))
        return xs, ys, es

    def plot_compare(metric_mean: str, metric_std: str, ylabel: str, title: str, filename: str):
        plt.figure()
        for total_qty in total_qty_grid:
            for algo in algos:
                xs, ys, es = series(metric_mean, metric_std, algo, total_qty)
                if not xs:
                    continue
                # Different marker by algo; legend includes both algo and qty.
                marker = "o" if algo == "TWAP" else "s"
                plt.errorbar(xs, ys, yerr=es, marker=marker, capsize=3, label=f"{algo} qty={total_qty}")

        plt.xscale("log")
        plt.xlabel("interval (events, log scale)  [TWAP=child_interval, VWAP=bucket_interval]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        path = out_dir / filename
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()

    plot_compare(
        "penalized_cost_per_share_mean",
        "penalized_cost_per_share_std",
        "penalized_cost_per_share (ticks per target share)",
        f"TWAP vs VWAP: penalized_cost_per_share vs interval (mean±95% CI, penalty={penalty_to_plot})",
        f"compare_penalized_vs_interval_pen{penalty_to_plot}_ci.png",
    )

    plot_compare(
        "shortfall_per_share_mean",
        "shortfall_per_share_std",
        "shortfall_per_share (ticks per filled share)",
        f"TWAP vs VWAP: shortfall_per_share vs interval (mean±95% CI, penalty={penalty_to_plot})",
        f"compare_shortfall_vs_interval_pen{penalty_to_plot}_ci.png",
    )

    plot_compare(
        "impact_per_share_mean",
        "impact_per_share_std",
        "impact_per_share (ticks beyond spread per filled share)",
        f"TWAP vs VWAP: impact_per_share vs interval (mean±95% CI, penalty={penalty_to_plot})",
        f"compare_impact_vs_interval_pen{penalty_to_plot}_ci.png",
    )

    print(
        f"\nSaved comparison plots to: {out_dir} (compare_*_pen{penalty_to_plot}_ci.png)"
    )
