from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

from experiments.run_truefx import run_truefx_strategy
from sim.flow import FlowConfig, HistoricalFormat
from sim.rl import ActorCritic, InventoryLiquidationEnv, LiquidationEnvConfig


def _shortfall_sell(arrival_mid: float, avg_px: float, filled_qty: float) -> float:
    return (arrival_mid - avg_px) * filled_qty


def _eval_rl(
    *,
    path: str,
    model: ActorCritic,
    device: str,
    env_cfg: LiquidationEnvConfig,
    flow_cfg: FlowConfig,
    episodes: int,
    stochastic: bool,
) -> Dict[str, float]:
    rewards: List[float] = []
    total_filled = 0.0
    total_notional = 0.0
    shortfalls: List[float] = []
    arrival_mids: List[float] = []

    for ep in range(episodes):
        cfg = replace(
            env_cfg,
            use_historical=True,
            historical_path=path,
            historical_paths=None,
            seed=env_cfg.seed + ep,
        )
        env = InventoryLiquidationEnv(cfg, flow_cfg=flow_cfg)
        obs = env.reset()
        obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)
        total_reward = 0.0
        done = False
        info: Dict[str, float] = {}
        while not done:
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                if stochastic:
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                else:
                    action = torch.argmax(logits, dim=-1).item()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)

        rewards.append(total_reward)
        episode_filled = float(info.get("episode_filled_qty", 0.0))
        episode_avg_px = float(info.get("episode_avg_px", 0.0))
        total_filled += episode_filled
        total_notional += episode_avg_px * episode_filled
        arrival_mid = info.get("arrival_mid")
        if arrival_mid is not None and not math.isnan(float(arrival_mid)):
            arrival_mids.append(float(arrival_mid))
        if arrival_mid is not None and not math.isnan(float(arrival_mid)) and episode_filled > 0:
            shortfalls.append(_shortfall_sell(float(arrival_mid), episode_avg_px, episode_filled))

    avg_fill_px = total_notional / total_filled if total_filled > 0 else 0.0
    filled_qty = total_filled / max(1, episodes)
    completion_rate = filled_qty / max(1.0, float(env_cfg.total_qty))
    unfilled_qty = max(0.0, float(env_cfg.total_qty) - filled_qty)
    arrival_mid_out = mean(arrival_mids) if arrival_mids else 0.0
    return {
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_median": median(rewards) if rewards else 0.0,
        "filled_qty": filled_qty,
        "avg_fill_px": avg_fill_px,
        "arrival_mid": arrival_mid_out,
        "shortfall": mean(shortfalls) if shortfalls else 0.0,
        "completion_rate": completion_rate,
        "unfilled_qty": unfilled_qty,
    }


def _compare_file(
    *,
    path: str,
    model: ActorCritic,
    device: str,
    env_cfg: LiquidationEnvConfig,
    flow_cfg: FlowConfig,
    episodes: int,
    stochastic: bool,
    child_interval_us: int,
    bucket_interval_us: int,
    fixed_qty: int,
    price_scale: int,
    warmup_ticks: int,
) -> List[Tuple[str, Dict[str, float]]]:
    rl_metrics = _eval_rl(
        path=path,
        model=model,
        device=device,
        env_cfg=env_cfg,
        flow_cfg=flow_cfg,
        episodes=episodes,
        stochastic=stochastic,
    )
    results = [("rl", rl_metrics)]

    for strat in ("twap", "vwap"):
        report = run_truefx_strategy(
            path=path,
            side="ASK",
            total_qty=env_cfg.total_qty,
            child_interval_us=child_interval_us,
            bucket_interval_us=bucket_interval_us,
            strategy=strat,
            fixed_qty=fixed_qty,
            price_scale=price_scale,
            warmup_ticks=warmup_ticks,
            min_child_qty=1,
            cleanup_unfilled=True,
            show_progress=False,
        )
        results.append(
            (
                strat,
                {
                    "reward_mean": 0.0,
                    "reward_median": 0.0,
                    "filled_qty": float(report.filled_qty),
                    "avg_fill_px": float(report.avg_fill_px or 0.0),
                    "arrival_mid": float(report.arrival_mid or 0.0),
                    "shortfall": float(report.shortfall or 0.0),
                    "completion_rate": float(report.completion_rate),
                    "unfilled_qty": float(report.unfilled_qty),
                },
            )
        )
    return results


def _worker_compare(args: Tuple[str, str, dict, int, bool, int, int, int, int, int]) -> List[Dict[str, float]]:
    (
        path,
        model_path,
        env_cfg_dict,
        episodes,
        stochastic,
        child_interval_us,
        bucket_interval_us,
        fixed_qty,
        price_scale,
        warmup_ticks,
    ) = args
    device = "cpu"
    fmt_map = {
        "truefx": HistoricalFormat.TRUEFX,
        "csv": HistoricalFormat.CSV,
        "binary": HistoricalFormat.BINARY,
    }
    fmt = fmt_map.get(env_cfg_dict.pop("historical_format_str", "truefx"), HistoricalFormat.TRUEFX)
    env_cfg = LiquidationEnvConfig(**env_cfg_dict, historical_format=fmt)
    flow_cfg = FlowConfig()
    model = ActorCritic(obs_dim=5, n_actions=env_cfg.action_bins).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = _compare_file(
        path=path,
        model=model,
        device=device,
        env_cfg=env_cfg,
        flow_cfg=flow_cfg,
        episodes=episodes,
        stochastic=stochastic,
        child_interval_us=child_interval_us,
        bucket_interval_us=bucket_interval_us,
        fixed_qty=fixed_qty,
        price_scale=price_scale,
        warmup_ticks=warmup_ticks,
    )
    
    rows: List[Dict[str, float]] = []
    for strat, metrics in results:
        row = {"file": Path(path).name, "strategy": strat}
        row.update(metrics)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RL vs TWAP/VWAP on TrueFX CSVs.")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV files.")
    parser.add_argument("--model-path", default="experiments/out/ppo_liquidation.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--total-qty", type=int, default=1000)
    parser.add_argument("--decision-interval-events", type=int, default=25)
    parser.add_argument("--action-bins", type=int, default=11)
    parser.add_argument("--penalty-per-share", type=float, default=2.0)
    parser.add_argument("--fixed-qty", type=int, default=1_000_000)
    parser.add_argument("--price-scale", type=int, default=100_000)
    parser.add_argument("--warmup-ticks", type=int, default=0)
    parser.add_argument("--child-interval-us", type=int, default=1_000_000)
    parser.add_argument("--bucket-interval-us", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    env_cfg = LiquidationEnvConfig(
        total_qty=args.total_qty,
        horizon_steps=50,
        decision_interval_events=args.decision_interval_events,
        action_bins=args.action_bins,
        warmup_events=0,
        penalty_per_share=args.penalty_per_share,
        seed=0,
        use_historical=True,
        historical_format=HistoricalFormat.TRUEFX,
        historical_fixed_qty=args.fixed_qty,
        historical_price_scale=args.price_scale,
        warmup_ticks=args.warmup_ticks,
    )
    flow_cfg = FlowConfig()

    model = None
    if args.workers == 1:
        model = ActorCritic(obs_dim=5, n_actions=env_cfg.action_bins).to(device)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

    data_dir = Path(args.data_dir)
    files = sorted(p for p in data_dir.glob("*.csv") if "rl_training" not in str(p))
    if not files:
        raise ValueError(f"no csv files found in {data_dir}")

    rows: List[Dict[str, float]] = []
    if args.workers == 1:
        for path in files:
            results = _compare_file(
                path=str(path),
                model=model,
                device=device,
                env_cfg=env_cfg,
                flow_cfg=flow_cfg,
                episodes=args.episodes,
                stochastic=args.stochastic,
                child_interval_us=args.child_interval_us,
                bucket_interval_us=args.bucket_interval_us,
                fixed_qty=args.fixed_qty,
                price_scale=args.price_scale,
                warmup_ticks=args.warmup_ticks,
            )
            for strat, metrics in results:
                row = {"file": path.name, "strategy": strat}
                row.update(metrics)
                rows.append(row)
    else:
        worker_count = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) // 2)
        env_cfg_dict = {
            "total_qty": env_cfg.total_qty,
            "horizon_steps": env_cfg.horizon_steps,
            "decision_interval_events": env_cfg.decision_interval_events,
            "action_bins": env_cfg.action_bins,
            "warmup_events": env_cfg.warmup_events,
            "penalty_per_share": env_cfg.penalty_per_share,
            "seed": env_cfg.seed,
            "use_historical": True,
            "historical_path": None,
            "historical_paths": None,
            "historical_format_str": "truefx",
            "historical_fixed_qty": env_cfg.historical_fixed_qty,
            "historical_price_scale": env_cfg.historical_price_scale,
            "warmup_ticks": env_cfg.warmup_ticks,
        }
        tasks = [
            (
                str(path),
                args.model_path,
                env_cfg_dict,
                args.episodes,
                args.stochastic,
                args.child_interval_us,
                args.bucket_interval_us,
                args.fixed_qty,
                args.price_scale,
                args.warmup_ticks,
            )
            for path in files
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_worker_compare, t) for t in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())

    for row in rows:
        print(
            f"{row['file']:>18} {row['strategy']:>4} "
            f"filled={row['filled_qty']:.0f} "
            f"comp={row['completion_rate']:.3f} "
            f"shortfall={row['shortfall']:.2f} "
            f"avg_px={row['avg_fill_px']:.2f}"
        )

    out_dir = Path("experiments/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_rl_twap_vwap.csv"
    with open(out_path, "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved results to {out_path}")

    # Plot comparison: shortfall by strategy per file.
    files_sorted = sorted({row["file"] for row in rows})
    strategies = ["rl", "twap", "vwap"]
    data = {s: [] for s in strategies}
    for fname in files_sorted:
        for strat in strategies:
            vals = [r["shortfall"] for r in rows if r["file"] == fname and r["strategy"] == strat]
            data[strat].append(vals[0] if vals else 0.0)

    x = range(len(files_sorted))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width for i in x], data["rl"], width=width, label="rl")
    ax.bar(x, data["twap"], width=width, label="twap")
    ax.bar([i + width for i in x], data["vwap"], width=width, label="vwap")
    ax.set_xticks(list(x))
    ax.set_xticklabels(files_sorted, rotation=45, ha="right")
    ax.set_ylabel("Shortfall")
    ax.set_title("RL vs TWAP vs VWAP (Shortfall)")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "compare_rl_twap_vwap.png"
    fig.savefig(plot_path)
    print(f"saved plot to {plot_path}")



if __name__ == "__main__":
    main()
