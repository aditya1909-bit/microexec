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


def _fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _mean_std(xs):
    # Population std (pstdev) to avoid NaNs for small n; with many seeds it's fine.
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), pstdev(xs)


def _run_one(seed: int, total_qty: int, child_interval: int, penalty_per_share: float, horizon_events: int, warmup_events: int) -> dict:
    # Create a fresh config per worker so parallel execution is clean.
    cfg = FlowConfig()

    book, rep = run_twap(
        side="BID",
        total_qty=total_qty,
        horizon_events=horizon_events,
        child_interval=child_interval,
        cfg=cfg,
        seed=seed,
        warmup_events=warmup_events,
        penalty_per_share=penalty_per_share,
    )

    d = asdict(rep)

    # "Impact" beyond crossing the spread (1 tick for BID market orders).
    sps = d.get("shortfall_per_share")
    if sps is None:
        d["impact_per_share"] = None
    else:
        d["impact_per_share"] = max(0.0, float(sps) - 1.0)

    bid_touch = book.depth(Side.BID, levels=1)
    ask_touch = book.depth(Side.ASK, levels=1)

    d.update(
        {
            "total_qty": total_qty,
            "child_interval": child_interval,
            "horizon_events": horizon_events,
            "seed": seed,
            "penalty_per_share": penalty_per_share,
            "end_bid_touch": bid_touch[0][1] if bid_touch else 0,
            "end_ask_touch": ask_touch[0][1] if ask_touch else 0,
            "end_imbalance_1": book.imbalance(levels=1),
        }
    )
    return d


if __name__ == "__main__":
    # Sweep parameters
    total_qty_grid = [250, 500, 1000, 2000]
    child_interval_grid = [50, 100, 200, 500, 1000]
    penalty_grid = [0.0, 1.0, 2.0, 5.0]

    # Seed averaging
    seed_grid = list(range(20))  # increase for smoother curves

    horizon_events = 10_000
    warmup_events = 500

    # Build task list
    tasks = []
    for seed in seed_grid:
        for total_qty in total_qty_grid:
            for child_interval in child_interval_grid:
                for penalty_per_share in penalty_grid:
                    tasks.append((seed, total_qty, child_interval, penalty_per_share))

    cpu = os.cpu_count() or 1
    chunk_size = max(1, len(tasks) // (cpu * 4))
    chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    rows = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        rows_chunk = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(seed, total_qty, child_interval, penalty_per_share, horizon_events, warmup_events)
            for seed, total_qty, child_interval, penalty_per_share in chunk
        )
        rows.extend(rows_chunk)
        progress_bar(idx, total_chunks, prefix="twap")

    #Aggregate by (total_qty, child_interval, penalty_per_share)
    grouped: dict[tuple[int, int, float], list[dict]] = {}
    for r in rows:
        key = (r["total_qty"], r["child_interval"], r["penalty_per_share"])
        grouped.setdefault(key, []).append(r)

    agg = []
    for (total_qty, child_interval, penalty_per_share), rs in grouped.items():
        pcs = [x["penalized_cost_per_share"] for x in rs if x["penalized_cost_per_share"] is not None]
        crs = [x["completion_rate"] for x in rs if x["completion_rate"] is not None]
        sps = [x["shortfall_per_share"] for x in rs if x["shortfall_per_share"] is not None]
        ips = [x["impact_per_share"] for x in rs if x.get("impact_per_share") is not None]

        pcs_m, pcs_s = _mean_std(pcs)
        cr_m, cr_s = _mean_std(crs)
        sp_m, sp_s = _mean_std(sps)
        ip_m, ip_s = _mean_std(ips)

        filled = [x["filled_qty"] for x in rs]
        unfilled = [x["unfilled_qty"] for x in rs]

        filled_m, filled_s = _mean_std(filled)
        unfilled_m, unfilled_s = _mean_std(unfilled)

        agg.append(
            {
                "total_qty": total_qty,
                "child_interval": child_interval,
                "penalty_per_share": penalty_per_share,
                "n_seeds": len(rs),
                "completion_rate_mean": cr_m,
                "completion_rate_std": cr_s,
                "penalized_cost_per_share_mean": pcs_m,
                "penalized_cost_per_share_std": pcs_s,
                "shortfall_per_share_mean": sp_m,
                "shortfall_per_share_std": sp_s,
                "impact_per_share_mean": ip_m,
                "impact_per_share_std": ip_s,
                "filled_qty_mean": filled_m,
                "filled_qty_std": filled_s,
                "unfilled_qty_mean": unfilled_m,
                "unfilled_qty_std": unfilled_s,
            }
        )

    def _key(a):
        v = a.get("penalized_cost_per_share_mean")
        return (v is None, v if v is not None else 1e18)

    agg_sorted = sorted(agg, key=_key)

    # --- Print summary (top 30 configs) ---
    cols = [
        "total_qty",
        "child_interval",
        "penalty_per_share",
        "n_seeds",
        "completion_rate_mean",
        "completion_rate_std",
        "penalized_cost_per_share_mean",
        "penalized_cost_per_share_std",
        "shortfall_per_share_mean",
        "shortfall_per_share_std",
        "impact_per_share_mean",
        "impact_per_share_std",
        "unfilled_qty_mean",
        "unfilled_qty_std",
    ]

    print("=== TWAP sweep summary (mean±std over seeds; sorted by penalized_cost_per_share_mean) ===")
    header = " | ".join([f"{c:>28}" for c in cols])
    print(header)
    print("-" * len(header))

    for a in agg_sorted[:30]:
        line = " | ".join([f"{_fmt(a.get(c)):>28}" for c in cols])
        print(line)

    #Plots with error bars
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    penalty_to_plot = 2.0

    def find_agg(total_qty: int, child_interval: int):
        return next(
            (
                a
                for a in agg
                if a["total_qty"] == total_qty
                and a["child_interval"] == child_interval
                and a["penalty_per_share"] == penalty_to_plot
            ),
            None,
        )

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available; skipping plots. ({e})")
        raise SystemExit(0)

    def ci95(std: float | None, n: int) -> float:
        if std is None or n <= 0:
            return 0.0
        return 1.96 * float(std) / sqrt(n)

    # Combined plot: penalized_cost_per_share vs child_interval (mean ± std) for all total_qty
    plt.figure()
    for total_qty in total_qty_grid:
        xs, ys, es = [], [], []
        for child_interval in child_interval_grid:
            a = find_agg(total_qty, child_interval)
            if a is None:
                continue
            y = a["penalized_cost_per_share_mean"]
            e = a["penalized_cost_per_share_std"]
            if y is None:
                continue
            xs.append(child_interval)
            ys.append(y)
            es.append(ci95(e, a["n_seeds"]))

        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"qty={total_qty}")

    plt.xscale("log")
    plt.xlabel("child_interval (events, log scale)")
    plt.ylabel("penalized_cost_per_share (ticks per target share)")
    plt.title(f"TWAP: penalized_cost_per_share vs child_interval (mean±95% CI, penalty={penalty_to_plot})")
    plt.legend()
    path = out_dir / f"twap_penalized_vs_interval_pen{penalty_to_plot}_ci.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

    # Combined plot: shortfall_per_share vs child_interval (mean ± std) for all total_qty
    plt.figure()
    for total_qty in total_qty_grid:
        xs, ys, es = [], [], []
        for child_interval in child_interval_grid:
            a = find_agg(total_qty, child_interval)
            if a is None:
                continue
            y = a["shortfall_per_share_mean"]
            e = a["shortfall_per_share_std"]
            if y is None:
                continue
            xs.append(child_interval)
            ys.append(y)
            es.append(ci95(e, a["n_seeds"]))

        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"qty={total_qty}")

    plt.xscale("log")
    plt.xlabel("child_interval (events, log scale)")
    plt.ylabel("shortfall_per_share (ticks per filled share)")
    plt.title(f"TWAP: shortfall_per_share vs child_interval (mean±95% CI, penalty={penalty_to_plot})")
    plt.legend()
    path = out_dir / f"twap_shortfall_vs_interval_pen{penalty_to_plot}_ci.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

    # Combined plot: impact_per_share vs child_interval (mean ± 95% CI) for all total_qty
    plt.figure()
    for total_qty in total_qty_grid:
        xs, ys, es = [], [], []
        for child_interval in child_interval_grid:
            a = find_agg(total_qty, child_interval)
            if a is None:
                continue
            y = a.get("impact_per_share_mean")
            e = a.get("impact_per_share_std")
            if y is None:
                continue
            xs.append(child_interval)
            ys.append(y)
            es.append(ci95(e, a["n_seeds"]))
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"qty={total_qty}")

    plt.xscale("log")
    plt.xlabel("child_interval (events, log scale)")
    plt.ylabel("impact_per_share (ticks beyond spread per filled share)")
    plt.title(f"TWAP: impact_per_share vs child_interval (mean±95% CI, penalty={penalty_to_plot})")
    plt.legend()
    path = out_dir / f"twap_impact_vs_interval_pen{penalty_to_plot}_ci.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

    print(
        f"\nSaved CI plots to: {out_dir} (twap_penalized_vs_interval_*_ci.png, twap_shortfall_vs_interval_*_ci.png, twap_impact_vs_interval_*_ci.png)"
    )
