from pathlib import Path
from statistics import mean, pstdev

from sim.flow import FlowConfig, PoissonOrderFlow, Side
from sim.execution.almgren_chriss import ImpactModel, run_almgren_chriss
from lob._cpp import lob_cpp
from experiments.progress import progress_bar


if __name__ == "__main__":
    cfg = FlowConfig()
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    plot_lambdas = [0.0, 1e-4, 1e-3, 1e-2]

    base_impact = ImpactModel(
        temporary_impact=0.02,
        permanent_impact=0.01,
        volatility=1.0,
        risk_aversion=0.0,
    )

    total_qty = 1000
    horizon_events = 10_000
    child_interval = 200
    warmup_events = 500
    seed = 0

    cap_abs = 40
    cap_frac = 0.05

    print("=== Almgren-Chriss lambda sweep ===")
    for idx, lam in enumerate(lambdas, start=1):
        impact = ImpactModel(
            temporary_impact=base_impact.temporary_impact,
            permanent_impact=base_impact.permanent_impact,
            volatility=base_impact.volatility,
            risk_aversion=lam,
        )

        book, rep = run_almgren_chriss(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=child_interval,
            cfg=cfg,
            impact=impact,
            seed=seed,
            warmup_events=warmup_events,
        )

        print(f"lambda={lam:g} filled={rep.filled_qty} avg_px={rep.avg_fill_px}")
        print(f"  shortfall={rep.shortfall} risk_penalty={rep.risk_penalty} objective={rep.objective}")
        print(f"  child_orders={rep.n_child_orders} first5={rep.child_qtys[:5]}")
        progress_bar(idx, len(lambdas), prefix="ac_lambda")

    # Plot schedules (uncapped)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_seconds = cfg.dt_us * child_interval / 1_000_000

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available; skipping plots. ({e})")
        raise SystemExit(0)

    plt.figure()
    for idx, lam in enumerate(plot_lambdas, start=1):
        impact = ImpactModel(
            temporary_impact=base_impact.temporary_impact,
            permanent_impact=base_impact.permanent_impact,
            volatility=base_impact.volatility,
            risk_aversion=lam,
        )
        _, rep = run_almgren_chriss(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=child_interval,
            cfg=cfg,
            impact=impact,
            seed=seed,
            warmup_events=warmup_events,
        )
        xs = [i * dt_seconds for i in range(len(rep.child_qtys))]
        plt.plot(xs, rep.child_qtys, label=f"λ={lam:g}")
        progress_bar(idx, len(plot_lambdas), prefix="ac_plot_uncapped")

    plt.xlabel("t (seconds)")
    plt.ylabel("q_t (shares)")
    plt.title("Almgren-Chriss schedules (uncapped)")
    plt.legend()
    plt.savefig(out_dir / "ac_schedules_uncapped.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Cap + objective change
    print("=== Per-slice cap impact ===")
    for idx, lam in enumerate((0.0, 1e-3, 1e-2), start=1):
        impact = ImpactModel(
            temporary_impact=base_impact.temporary_impact,
            permanent_impact=base_impact.permanent_impact,
            volatility=base_impact.volatility,
            risk_aversion=lam,
        )
        _, rep_capped = run_almgren_chriss(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=child_interval,
            cfg=cfg,
            impact=impact,
            cap_abs=cap_abs,
            cap_frac=cap_frac,
            seed=seed,
            warmup_events=warmup_events,
        )
        print(
            f"lambda={lam:g} cap_used={rep_capped.cap_used} "
            f"objective_uncapped={rep_capped.objective_uncapped} objective_capped={rep_capped.objective}"
        )
        progress_bar(idx, 3, prefix="ac_cap")

    plt.figure()
    for idx, lam in enumerate(plot_lambdas, start=1):
        impact = ImpactModel(
            temporary_impact=base_impact.temporary_impact,
            permanent_impact=base_impact.permanent_impact,
            volatility=base_impact.volatility,
            risk_aversion=lam,
        )
        _, rep = run_almgren_chriss(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=child_interval,
            cfg=cfg,
            impact=impact,
            cap_abs=cap_abs,
            cap_frac=cap_frac,
            seed=seed,
            warmup_events=warmup_events,
        )
        xs = [i * dt_seconds for i in range(len(rep.child_qtys))]
        plt.plot(xs, rep.child_qtys, label=f"λ={lam:g}")
        progress_bar(idx, len(plot_lambdas), prefix="ac_plot_capped")

    plt.xlabel("t (seconds)")
    plt.ylabel("q_t (shares)")
    plt.title(f"Almgren-Chriss schedules (cap_abs={cap_abs}, cap_frac={cap_frac})")
    plt.legend()
    plt.savefig(out_dir / "ac_schedules_capped.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Compare AC vs TWAP/VWAP on shared seeds/regimes.
    print("=== AC vs TWAP vs VWAP (mean±std) ===")
    seeds = list(range(20))
    compare_lambdas = [0.0, 1e-3, 1e-2]
    bucket_interval = child_interval
    dt_seconds_bucket = cfg.dt_us * bucket_interval / 1_000_000
    pov_rate = 0.1

    def _mean_std(xs):
        if not xs:
            return None, None
        if len(xs) == 1:
            return xs[0], 0.0
        return mean(xs), pstdev(xs)

    def _compute_bucket_targets(total_qty: int, vols: list[int]) -> list[int]:
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
        fracs = [(raw[i] - floors[i], i) for i in range(n)]
        fracs.sort(reverse=True)
        out = floors[:]
        for k in range(rem):
            out[fracs[k][1]] += 1
        s = sum(out)
        if s != total_qty:
            out[-1] += total_qty - s
        return out

    def _forecast_market_volume(seed_val: int) -> list[int]:
        book = lob_cpp.LimitOrderBook()
        flow = PoissonOrderFlow(cfg, seed=seed_val)
        ts_local = 0
        for _ in range(warmup_events):
            ts_local += cfg.dt_us
            flow.step(book, ts_local)
        n_buckets = max(1, (horizon_events + bucket_interval - 1) // bucket_interval)
        vols = [0] * n_buckets
        for i in range(horizon_events):
            ts_local += cfg.dt_us
            kind, traded = flow.step(book, ts_local)
            if kind == lob_cpp.EventKind.MARKET:
                vols[i // bucket_interval] += int(traded)
        return vols

    def _pov_schedule(total_qty: int, vols: list[int], rate: float) -> list[int]:
        if rate <= 0.0:
            return [0 for _ in vols]
        if not vols:
            return []
        schedule = []
        remaining = total_qty
        for v in vols:
            qty = int(round(rate * v))
            qty = min(remaining, max(0, qty))
            schedule.append(qty)
            remaining -= qty
            if remaining <= 0:
                break
        if remaining > 0:
            schedule.extend([0] * (len(vols) - len(schedule)))
            i = 0
            while remaining > 0 and schedule:
                schedule[i] += 1
                remaining -= 1
                i = (i + 1) % len(schedule)
        return schedule

    def _simulate_schedule(seed_val: int, schedule: list[int], interval: int) -> float | None:
        book = lob_cpp.LimitOrderBook()
        flow = PoissonOrderFlow(cfg, seed=seed_val)
        ts_local = 0
        for _ in range(warmup_events):
            ts_local += cfg.dt_us
            flow.step(book, ts_local)

        arrival_mid = book.mid()
        if arrival_mid is None:
            return None

        notional = 0
        filled = 0
        slice_idx = 0

        for i in range(horizon_events):
            ts_local += cfg.dt_us
            flow.step(book, ts_local)

            if (i % interval) == 0 and slice_idx < len(schedule):
                qty = schedule[slice_idx]
                slice_idx += 1
                if qty > 0:
                    fills = book.add_market(Side.BID, qty, ts_local)
                    for f in fills:
                        notional += f.qty * f.px
                        filled += f.qty

        if filled == 0:
            return None
        avg_px = notional / filled
        return (avg_px - arrival_mid) * filled

    def _risk_penalty(schedule: list[int], lam: float, dt: float) -> float:
        if lam <= 0:
            return 0.0
        x = float(total_qty)
        penalty = 0.0
        for q in schedule:
            penalty += lam * base_impact.volatility * base_impact.volatility * x * x * dt
            x -= float(q)
        return penalty

    def _twap_schedule() -> list[int]:
        n_slices = max(1, (horizon_events + child_interval - 1) // child_interval)
        base = total_qty // n_slices
        rem = total_qty - base * n_slices
        return [base + (1 if i < rem else 0) for i in range(n_slices)]

    def _vwap_schedule(seed_val: int) -> list[int]:
        vols = _forecast_market_volume(seed_val)
        return _compute_bucket_targets(total_qty, vols)

    def _ac_schedule(lam: float) -> list[int]:
        impact = ImpactModel(
            temporary_impact=base_impact.temporary_impact,
            permanent_impact=base_impact.permanent_impact,
            volatility=base_impact.volatility,
            risk_aversion=lam,
        )
        _, rep = run_almgren_chriss(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=child_interval,
            cfg=cfg,
            impact=impact,
            seed=seed,
            warmup_events=warmup_events,
        )
        return list(rep.child_qtys)

    def _ac_schedule_capped(lam: float) -> list[int]:
        impact = ImpactModel(
            temporary_impact=base_impact.temporary_impact,
            permanent_impact=base_impact.permanent_impact,
            volatility=base_impact.volatility,
            risk_aversion=lam,
        )
        _, rep = run_almgren_chriss(
            side="BID",
            total_qty=total_qty,
            horizon_events=horizon_events,
            child_interval=child_interval,
            cfg=cfg,
            impact=impact,
            cap_abs=cap_abs,
            cap_frac=cap_frac,
            seed=seed,
            warmup_events=warmup_events,
        )
        return list(rep.child_qtys)

    for idx_lam, lam in enumerate(compare_lambdas, start=1):
        ac_shortfalls = []
        ac_objectives = []
        twap_shortfalls = []
        twap_objectives = []
        vwap_shortfalls = []
        vwap_objectives = []
        pov_shortfalls = []
        pov_objectives = []

        for s in seeds:
            ac_schedule = _ac_schedule(lam)
            ac_sf = _simulate_schedule(s, ac_schedule, child_interval)
            if ac_sf is not None:
                ac_shortfalls.append(ac_sf)
                ac_objectives.append(ac_sf + _risk_penalty(ac_schedule, lam, dt_seconds))

            twap_schedule = _twap_schedule()
            twap_sf = _simulate_schedule(s, twap_schedule, child_interval)
            if twap_sf is not None:
                twap_shortfalls.append(twap_sf)
                twap_objectives.append(twap_sf + _risk_penalty(twap_schedule, lam, dt_seconds))

            vwap_schedule = _vwap_schedule(s)
            vwap_sf = _simulate_schedule(s, vwap_schedule, bucket_interval)
            if vwap_sf is not None:
                vwap_shortfalls.append(vwap_sf)
                vwap_objectives.append(vwap_sf + _risk_penalty(vwap_schedule, lam, dt_seconds_bucket))

            pov_schedule = _pov_schedule(total_qty, _forecast_market_volume(s), pov_rate)
            pov_sf = _simulate_schedule(s, pov_schedule, bucket_interval)
            if pov_sf is not None:
                pov_shortfalls.append(pov_sf)
                pov_objectives.append(pov_sf + _risk_penalty(pov_schedule, lam, dt_seconds_bucket))

        def _fmt(m, s):
            if m is None:
                return "None"
            if s is None:
                return f"{m:.4f}"
            return f"{m:.4f}±{s:.4f}"

        ac_sf_m, ac_sf_s = _mean_std([x for x in ac_shortfalls if x is not None])
        ac_obj_m, ac_obj_s = _mean_std([x for x in ac_objectives if x is not None])
        twap_sf_m, twap_sf_s = _mean_std([x for x in twap_shortfalls if x is not None])
        twap_obj_m, twap_obj_s = _mean_std([x for x in twap_objectives if x is not None])
        vwap_sf_m, vwap_sf_s = _mean_std([x for x in vwap_shortfalls if x is not None])
        vwap_obj_m, vwap_obj_s = _mean_std([x for x in vwap_objectives if x is not None])
        pov_sf_m, pov_sf_s = _mean_std([x for x in pov_shortfalls if x is not None])
        pov_obj_m, pov_obj_s = _mean_std([x for x in pov_objectives if x is not None])

        print(f"lambda={lam:g}")
        print(f"  AC   shortfall={_fmt(ac_sf_m, ac_sf_s)} objective={_fmt(ac_obj_m, ac_obj_s)}")
        print(f"  TWAP shortfall={_fmt(twap_sf_m, twap_sf_s)} objective={_fmt(twap_obj_m, twap_obj_s)}")
        print(f"  VWAP shortfall={_fmt(vwap_sf_m, vwap_sf_s)} objective={_fmt(vwap_obj_m, vwap_obj_s)}")
        print(f"  POV  shortfall={_fmt(pov_sf_m, pov_sf_s)} objective={_fmt(pov_obj_m, pov_obj_s)}")
        progress_bar(idx_lam, len(compare_lambdas), prefix="ac_compare")

    # Efficient frontier: std(shortfall) vs mean(shortfall)
    ac_frontier = []
    for idx, lam in enumerate(lambdas, start=1):
        ac_schedule = _ac_schedule(lam)
        sfs = []
        for s in seeds:
            sf = _simulate_schedule(s, ac_schedule, child_interval)
            if sf is not None:
                sfs.append(sf)
        m, sdev = _mean_std(sfs)
        if m is not None and sdev is not None:
            ac_frontier.append((lam, m, sdev))
        progress_bar(idx, len(lambdas), prefix="ac_frontier")

    ac_frontier_capped = []
    for idx, lam in enumerate(lambdas, start=1):
        ac_schedule = _ac_schedule_capped(lam)
        sfs = []
        for s in seeds:
            sf = _simulate_schedule(s, ac_schedule, child_interval)
            if sf is not None:
                sfs.append(sf)
        m, sdev = _mean_std(sfs)
        if m is not None and sdev is not None:
            ac_frontier_capped.append((lam, m, sdev))
        progress_bar(idx, len(lambdas), prefix="ac_frontier_capped")

    twap_schedule = _twap_schedule()
    twap_sfs = []
    for s in seeds:
        sf = _simulate_schedule(s, twap_schedule, child_interval)
        if sf is not None:
            twap_sfs.append(sf)
    twap_m, twap_s = _mean_std(twap_sfs)

    vwap_sfs = []
    for s in seeds:
        sf = _simulate_schedule(s, _vwap_schedule(s), bucket_interval)
        if sf is not None:
            vwap_sfs.append(sf)
    vwap_m, vwap_s = _mean_std(vwap_sfs)

    pov_sfs = []
    for s in seeds:
        sf = _simulate_schedule(s, _pov_schedule(total_qty, _forecast_market_volume(s), pov_rate), bucket_interval)
        if sf is not None:
            pov_sfs.append(sf)
    pov_m, pov_s = _mean_std(pov_sfs)

    plt.figure()
    if ac_frontier:
        xs = [v[2] for v in ac_frontier]
        ys = [v[1] for v in ac_frontier]
        plt.plot(xs, ys, marker="o", label="AC (lambda sweep)")
        for lam, mean_sf, std_sf in ac_frontier:
            plt.annotate(f"{lam:g}", (std_sf, mean_sf))

    if ac_frontier_capped:
        xs = [v[2] for v in ac_frontier_capped]
        ys = [v[1] for v in ac_frontier_capped]
        plt.plot(xs, ys, marker="o", label="AC (capped)")

    if twap_m is not None and twap_s is not None:
        plt.scatter([twap_s], [twap_m], marker="s", label="TWAP")
    if vwap_m is not None and vwap_s is not None:
        plt.scatter([vwap_s], [vwap_m], marker="^", label="VWAP")
    if pov_m is not None and pov_s is not None:
        plt.scatter([pov_s], [pov_m], marker="D", label=f"POV (rate={pov_rate})")

    plt.xlabel("std(shortfall)")
    plt.ylabel("mean(shortfall)")
    plt.title("Efficient frontier: AC (uncapped/capped) vs TWAP/VWAP/POV")
    plt.legend()
    plt.savefig(out_dir / "ac_efficient_frontier.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Capped vs uncapped AC at lambda=1e-2
    lam_focus = 1e-2
    uncapped = None
    capped = None
    for lam, mean_sf, std_sf in ac_frontier:
        if abs(lam - lam_focus) < 1e-12:
            uncapped = (mean_sf, std_sf)
            break
    for lam, mean_sf, std_sf in ac_frontier_capped:
        if abs(lam - lam_focus) < 1e-12:
            capped = (mean_sf, std_sf)
            break
    print("=== AC cap shift (lambda=1e-2) ===")
    print(f"  uncapped mean={uncapped[0] if uncapped else None} std={uncapped[1] if uncapped else None}")
    print(f"  capped   mean={capped[0] if capped else None} std={capped[1] if capped else None}")
