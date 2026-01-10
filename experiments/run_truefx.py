from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from queue import Empty
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from lob.book import LimitOrderBook
from lob.types import Side
from sim.flow import HistoricalFormat, HistoricalOrderFlow
from experiments.progress import progress_bar


@dataclass
class Report:
    side: str
    target_qty: int
    filled_qty: int
    avg_fill_px: float | None
    arrival_mid: float | None
    shortfall: float | None
    n_child_orders: int
    unfilled_qty: int
    completion_rate: float


def parse_truefx_ts_us(token: str) -> int:
    dt = datetime.strptime(token.strip(), "%Y%m%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def scan_truefx(path: str, bucket_interval_us: int) -> tuple[int, int, list[int], int]:
    start_ts = None
    end_ts = None
    buckets: list[int] = []
    valid_rows = 0
    if bucket_interval_us <= 0:
        bucket_interval_us = 1_000_000
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                ts = parse_truefx_ts_us(parts[1])
            except ValueError:
                continue
            if start_ts is None:
                start_ts = ts
            end_ts = ts
            idx = (ts - start_ts) // bucket_interval_us
            while len(buckets) <= idx:
                buckets.append(0)
            buckets[idx] += 1
            valid_rows += 1
    if start_ts is None or end_ts is None:
        raise ValueError("No valid TrueFX rows found")
    return start_ts, end_ts, buckets, valid_rows


def compute_bucket_targets(total_qty: int, buckets: Iterable[int]) -> list[int]:
    vols = list(buckets)
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
    fracs = sorted(((raw[i] - floors[i], i) for i in range(n)), reverse=True)
    out = floors[:]
    for i in range(rem):
        out[fracs[i][1]] += 1
    return out


def shortfall(side: str, arrival_mid: float, avg_px: float, filled_qty: int) -> float:
    if side == "BID":
        return (avg_px - arrival_mid) * filled_qty
    return (arrival_mid - avg_px) * filled_qty


def run_truefx_strategy(
    *,
    path: str,
    side: str,
    total_qty: int,
    child_interval_us: int,
    bucket_interval_us: int,
    strategy: str,
    fixed_qty: int,
    price_scale: int,
    warmup_ticks: int,
    min_child_qty: int,
    cleanup_unfilled: bool,
    show_progress: bool = True,
    progress_queue: mp.Queue | None = None,
) -> Report:
    start_ts, end_ts, bucket_counts, total_rows = scan_truefx(path, bucket_interval_us)
    total_duration_us = max(1, end_ts - start_ts)
    send_interval_us = bucket_interval_us if strategy == "vwap" else child_interval_us
    n_slices = max(1, (total_duration_us + send_interval_us - 1) // send_interval_us)
    if min_child_qty > 0:
        max_slices = max(1, total_qty // min_child_qty)
        n_slices = min(n_slices, max_slices)
    bucket_targets = compute_bucket_targets(total_qty, bucket_counts)

    book = LimitOrderBook()
    flow = HistoricalOrderFlow(
        path,
        HistoricalFormat.TRUEFX,
        fixed_qty=fixed_qty,
        price_scale=price_scale,
    )

    ts = 0
    for _ in range(warmup_ticks):
        ok, ts, _ = flow.step(book)
        if not ok:
            break

    arrival_mid = book.mid()
    if arrival_mid is None:
        ok, ts, _ = flow.step(book)
        if ok:
            arrival_mid = book.mid()

    remaining = total_qty
    filled_qty = 0
    notional = 0
    n_child = 0
    next_send_ts = None
    bucket_idx = 0
    carry_target = 0
    last_ts_seen = None

    step_count = 0
    progress_every = max(1, total_rows // 200)
    while remaining > 0:
        ok, ts, _ = flow.step(book)
        if not ok:
            break
        step_count += 1
        last_ts_seen = ts
        if show_progress and (step_count % progress_every == 0 or step_count == total_rows):
            if progress_queue is not None:
                progress_queue.put((strategy, step_count, total_rows))
            else:
                progress_bar(step_count, total_rows, prefix=strategy)
        if next_send_ts is None:
            next_send_ts = ts
        if ts < next_send_ts:
            continue

        if strategy == "vwap":
            target = bucket_targets[bucket_idx] if bucket_idx < len(bucket_targets) else remaining
            carry_target += max(0, target)
            if min_child_qty > 0 and carry_target < min_child_qty and remaining > min_child_qty:
                next_send_ts += send_interval_us
                bucket_idx += 1
                continue
            child_qty = min(remaining, carry_target)
            carry_target = 0
        else:
            slices_left = max(1, n_slices - (bucket_idx))
            child_qty = (remaining + slices_left - 1) // slices_left
            if min_child_qty > 0:
                child_qty = max(min_child_qty, child_qty)
            child_qty = min(remaining, child_qty)

        if child_qty > 0:
            n_child += 1
            side_enum = Side.BID if side == "BID" else Side.ASK
            fills = book.add_market(side_enum, child_qty, ts)
            for f in fills:
                filled_qty += f.qty
                notional += f.qty * f.px
            remaining = max(0, total_qty - filled_qty)

        next_send_ts += send_interval_us
        bucket_idx += 1

    if cleanup_unfilled and remaining > 0 and last_ts_seen is not None:
        side_enum = Side.BID if side == "BID" else Side.ASK
        fills = book.add_market(side_enum, remaining, last_ts_seen)
        for f in fills:
            filled_qty += f.qty
            notional += f.qty * f.px
        remaining = max(0, total_qty - filled_qty)

    if show_progress:
        if progress_queue is not None:
            progress_queue.put((strategy, min(step_count, total_rows), total_rows))
        else:
            progress_bar(min(step_count, total_rows), total_rows, prefix=strategy)

    avg_px = (notional / filled_qty) if filled_qty > 0 else None
    sf = None
    if arrival_mid is not None and avg_px is not None:
        sf = shortfall(side, arrival_mid, avg_px, filled_qty)

    unfilled_qty = max(0, total_qty - filled_qty)
    completion_rate = filled_qty / total_qty if total_qty > 0 else 0.0

    return Report(
        side=side,
        target_qty=total_qty,
        filled_qty=filled_qty,
        avg_fill_px=avg_px,
        arrival_mid=arrival_mid,
        shortfall=sf,
        n_child_orders=n_child,
        unfilled_qty=unfilled_qty,
        completion_rate=completion_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple strategies on TrueFX BBO data.")
    parser.add_argument(
        "--path",
        default="data/EURUSD-2025-07.csv",
        help="Path to TrueFX CSV file",
    )
    parser.add_argument("--side", choices=["BID", "ASK"], default="BID")
    parser.add_argument("--total-qty", type=int, default=100_000)
    parser.add_argument("--strategy", choices=["twap", "vwap", "both"], default="both")
    parser.add_argument("--child-interval-us", type=int, default=1_000_000)
    parser.add_argument("--bucket-interval-us", type=int, default=1_000_000)
    parser.add_argument("--fixed-qty", type=int, default=1_000_000)
    parser.add_argument("--price-scale", type=int, default=100_000)
    parser.add_argument("--warmup-ticks", type=int, default=0)
    parser.add_argument("--min-child-qty", type=int, default=100)
    parser.add_argument("--no-cleanup", action="store_true", help="Disable end-of-file cleanup fill")
    args = parser.parse_args()

    strategies = ["twap", "vwap"] if args.strategy == "both" else [args.strategy]
    reports = []
    if args.strategy == "both":
        manager = mp.Manager()
        progress_queue = manager.Queue()
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_truefx_strategy,
                    path=args.path,
                    side=args.side,
                    total_qty=args.total_qty,
                    child_interval_us=args.child_interval_us,
                    bucket_interval_us=args.bucket_interval_us,
                    strategy=strat,
                    fixed_qty=args.fixed_qty,
                    price_scale=args.price_scale,
                    warmup_ticks=args.warmup_ticks,
                    min_child_qty=args.min_child_qty,
                    cleanup_unfilled=not args.no_cleanup,
                    show_progress=True,
                    progress_queue=progress_queue,
                ): strat
                for strat in strategies
            }
            done = 0
            last_seen: dict[str, tuple[int, int]] = {}
            while done < len(futures):
                try:
                    strat, cur, total = progress_queue.get(timeout=0.2)
                    last_seen[strat] = (cur, total)
                    left = " | ".join(
                        f"{k} {v[0]}/{v[1]} ({(v[0] / v[1] * 100.0):.1f}%)"
                        for k, v in sorted(last_seen.items())
                    )
                    elapsed = time.perf_counter() - start_time
                    sys.stdout.write(f"\r{left} | elapsed {elapsed:6.1f}s")
                    sys.stdout.flush()
                except Empty:
                    pass
                done = sum(1 for f in futures if f.done())

            while True:
                try:
                    strat, cur, total = progress_queue.get_nowait()
                    last_seen[strat] = (cur, total)
                    left = " | ".join(
                        f"{k} {v[0]}/{v[1]} ({(v[0] / v[1] * 100.0):.1f}%)"
                        for k, v in sorted(last_seen.items())
                    )
                    elapsed = time.perf_counter() - start_time
                    sys.stdout.write(f"\r{left} | elapsed {elapsed:6.1f}s")
                    sys.stdout.flush()
                except Empty:
                    break
            if last_seen:
                sys.stdout.write("\n")
                sys.stdout.flush()

            for future in futures:
                strat = futures[future]
                report = future.result()
                reports.append((strat, report))
    else:
        start_time = time.perf_counter()
        for strat in strategies:
            report = run_truefx_strategy(
                path=args.path,
                side=args.side,
                total_qty=args.total_qty,
                child_interval_us=args.child_interval_us,
                bucket_interval_us=args.bucket_interval_us,
                strategy=strat,
                fixed_qty=args.fixed_qty,
                price_scale=args.price_scale,
                warmup_ticks=args.warmup_ticks,
                min_child_qty=args.min_child_qty,
                cleanup_unfilled=not args.no_cleanup,
                show_progress=True,
            )
            elapsed = time.perf_counter() - start_time
            sys.stdout.write(f"\r{strat} 100.0% | elapsed {elapsed:6.1f}s\n")
            reports.append((strat, report))

    for strat, report in reports:
        print("strategy:", strat)
        print("side:", report.side)
        print("target_qty:", report.target_qty)
        print("filled_qty:", report.filled_qty)
        print("avg_fill_px:", report.avg_fill_px)
        print("arrival_mid:", report.arrival_mid)
        print("shortfall:", report.shortfall)
        print("n_child_orders:", report.n_child_orders)
        print("unfilled_qty:", report.unfilled_qty)
        print("completion_rate:", report.completion_rate)
        print("---")

    if len(reports) == 2:
        (s1, r1), (s2, r2) = reports
        sf1 = r1.shortfall if r1.shortfall is not None else 0.0
        sf2 = r2.shortfall if r2.shortfall is not None else 0.0
        print("comparison:", f"{s1} vs {s2}")
        print("shortfall_delta:", sf1 - sf2)
        print("filled_qty_delta:", r1.filled_qty - r2.filled_qty)
        print("avg_fill_px_delta:", (r1.avg_fill_px or 0.0) - (r2.avg_fill_px or 0.0))


if __name__ == "__main__":
    main()
