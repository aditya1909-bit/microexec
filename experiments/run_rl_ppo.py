from __future__ import annotations

import argparse
from collections import deque
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from experiments.progress import progress_bar
from sim.flow import FlowConfig, HistoricalFormat
from sim.rl import InventoryLiquidationEnv, LiquidationEnvConfig, PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-path", type=str, default=None)
    parser.add_argument("--historical-dir", type=str, default="data/rl_training/September")
    parser.add_argument("--historical-format", type=str, default="truefx")
    parser.add_argument("--fixed-qty", type=int, default=1_000_000)
    parser.add_argument("--price-scale", type=int, default=100_000)
    parser.add_argument("--warmup-ticks", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--total-qty", type=int, default=1000)
    parser.add_argument("--horizon-steps", type=int, default=50)
    parser.add_argument("--decision-interval", type=int, default=25)
    parser.add_argument("--action-bins", type=int, default=11)
    parser.add_argument("--penalty-per-share", type=float, default=2.0)
    parser.add_argument("--total-updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--mp-envs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-envs", type=int, default=-1)
    parser.add_argument("--num-threads", type=int, default=-1)
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

    env_cfg = LiquidationEnvConfig(
        total_qty=args.total_qty,
        horizon_steps=args.horizon_steps,
        decision_interval_events=args.decision_interval,
        action_bins=args.action_bins,
        warmup_events=500,
        penalty_per_share=args.penalty_per_share,
        seed=0,
        use_historical=args.historical_path is not None or historical_paths is not None,
        historical_path=args.historical_path,
        historical_paths=historical_paths,
        historical_format=fmt_map[fmt_key],
        historical_fixed_qty=args.fixed_qty,
        historical_price_scale=args.price_scale,
        warmup_ticks=args.warmup_ticks,
    )
    cpu_count = os.cpu_count() or 8
    num_envs = args.num_envs if args.num_envs > 0 else cpu_count * 2
    num_threads = args.num_threads if args.num_threads > 0 else cpu_count
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))

    ppo_cfg = PPOConfig(
        total_updates=args.total_updates,
        rollout_steps=args.rollout_steps,
        num_envs=num_envs,
        learning_rate=3e-4,
        minibatch_size=256,
        update_epochs=4,
        seed=0,
        use_mp_envs=args.mp_envs,
    )
    flow_cfg = None if args.mp_envs else FlowConfig()

    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print(f"device={device} num_envs={num_envs} num_threads={num_threads}")
    roll_window = 10
    roll = deque(maxlen=roll_window)

    avg_history: list[float] = []

    start_time = time.monotonic()

    def _fmt_mmss(seconds: float) -> str:
        seconds_int = max(0, int(round(seconds)))
        mins, secs = divmod(seconds_int, 60)
        return f"{mins:02d}:{secs:02d}"

    def _progress(cur: int, total: int, avg: float) -> None:
        roll.append(avg)
        roll_avg = sum(roll) / len(roll)
        elapsed = max(1e-9, time.monotonic() - start_time)
        rate = cur / elapsed
        remaining = max(0.0, total - cur) / max(rate, 1e-9)
        progress_bar(
            cur,
            total,
            prefix=f"ppo avg_return={avg:.4f} roll{roll_window}={roll_avg:.4f} eta={_fmt_mmss(remaining)}",
        )
        avg_history.append(avg)

    def _rollout_progress(cur: int, total: int) -> None:
        progress_bar(cur, total, prefix="rollout")

    def _opt_progress(cur: int, total: int) -> None:
        progress_bar(cur, total, prefix="opt")

    model = train_ppo(
        env_cfg=env_cfg,
        flow_cfg=flow_cfg,
        ppo_cfg=ppo_cfg,
        device=device,
        progress_fn=_progress,
        rollout_progress_fn=_rollout_progress,
        opt_progress_fn=_opt_progress,
    )

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_liquidation.pt"
    torch.save(model.state_dict(), model_path)
    print(f"saved model to {model_path}")
    if avg_history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(avg_history) + 1), avg_history, label="avg_return")
        ax.set_title("PPO Training Curve")
        ax.set_xlabel("Update")
        ax.set_ylabel("Avg Return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        curve_path = out_dir / "ppo_training_curve.png"
        fig.savefig(curve_path)
        print(f"saved training curve to {curve_path}")

    # Stochastic eval across multiple episodes.
    eval_episodes = 10
    eval_rewards: list[float] = []
    eval_infos: list[dict] = []
    for _ in range(eval_episodes):
        eval_flow_cfg = FlowConfig() if flow_cfg is None else flow_cfg
        env = InventoryLiquidationEnv(env_cfg, flow_cfg=eval_flow_cfg)
        obs = env.reset()
        obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)

        total_reward = 0.0
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=device)

        eval_rewards.append(total_reward)
        eval_infos.append(info)

    eval_rewards_sorted = sorted(eval_rewards)
    mid = len(eval_rewards_sorted) // 2
    if len(eval_rewards_sorted) % 2 == 0:
        median_reward = 0.5 * (eval_rewards_sorted[mid - 1] + eval_rewards_sorted[mid])
    else:
        median_reward = eval_rewards_sorted[mid]
    mean_reward = sum(eval_rewards) / len(eval_rewards)
    fill_ratios = []
    for info in eval_infos:
        filled = info.get("episode_filled_qty")
        if filled is None:
            continue
        fill_ratios.append(float(filled) / max(1.0, float(env_cfg.total_qty)))
    fill_mean = sum(fill_ratios) / len(fill_ratios) if fill_ratios else 0.0
    fill_sorted = sorted(fill_ratios)
    mid_fill = len(fill_sorted) // 2
    if fill_sorted:
        if len(fill_sorted) % 2 == 0:
            fill_median = 0.5 * (fill_sorted[mid_fill - 1] + fill_sorted[mid_fill])
        else:
            fill_median = fill_sorted[mid_fill]
    else:
        fill_median = 0.0

    print("eval_mean_reward", mean_reward)
    print("eval_median_reward", median_reward)
    print("eval_fill_ratio_mean", fill_mean)
    print("eval_fill_ratio_median", fill_median)
    print("eval_last_info", eval_infos[-1] if eval_infos else {})


if __name__ == "__main__":
    main()
