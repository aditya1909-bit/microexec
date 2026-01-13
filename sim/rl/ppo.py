from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import multiprocessing as mp
import time
import math
import random
import sys
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from sim.rl.env import InventoryLiquidationEnv, LiquidationEnvConfig


@dataclass
class PPOConfig:
    total_updates: int = 200
    rollout_steps: int = 256
    num_envs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    minibatch_size: int = 256
    update_epochs: int = 4
    seed: int = 0
    use_mp_envs: bool = False


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        hidden = 128
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        value = self.value(obs).squeeze(-1)
        return logits, value


def train_ppo(
    *,
    env_cfg: LiquidationEnvConfig,
    flow_cfg=None,
    ppo_cfg: PPOConfig,
    device: str = "cpu",
    progress_fn: Optional[Callable[[int, int, float], None]] = None,
    rollout_progress_fn: Optional[Callable[[int, int], None]] = None,
    opt_progress_fn: Optional[Callable[[int, int], None]] = None,
) -> ActorCritic:
    torch.manual_seed(ppo_cfg.seed)
    random.seed(ppo_cfg.seed)

    env_pool = None
    if ppo_cfg.use_mp_envs:
        if flow_cfg is not None:
            raise ValueError("use_mp_envs requires flow_cfg=None (workers create default FlowConfig).")
        env_pool = _AsyncEnvPool(env_cfg, ppo_cfg.num_envs)
        obs = env_pool.obs
        envs = None
    else:
        envs = [
            InventoryLiquidationEnv(replace(env_cfg, seed=env_cfg.seed + i), flow_cfg=flow_cfg)
            for i in range(ppo_cfg.num_envs)
        ]
        obs = [env.reset() for env in envs]

    if ppo_cfg.use_mp_envs:
        obs_dim = len(obs[0])
        n_actions = env_cfg.action_bins
    else:
        obs_dim = envs[0].obs_dim
        n_actions = envs[0].n_actions

    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)

    try:
        for update in range(ppo_cfg.total_updates):
            update_start = time.monotonic()
            batch_obs = []
            batch_actions = []
            batch_logprobs = []
            batch_rewards = []
            batch_dones = []
            batch_values = []

            for _ in range(ppo_cfg.rollout_steps):
                if update == 0 and rollout_progress_fn is not None:
                    rollout_progress_fn(len(batch_obs) + 1, ppo_cfg.rollout_steps)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                logits, values = model(obs_tensor)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                if ppo_cfg.use_mp_envs:
                    next_obs, rewards, dones = env_pool.step(actions.tolist())
                else:
                    next_obs = []
                    rewards = []
                    dones = []
                    for i, env in enumerate(envs):
                        o, r, done, _info = env.step(int(actions[i].item()))
                        if done:
                            o = env.reset()
                        next_obs.append(o)
                        rewards.append(float(r))
                        dones.append(float(done))

                batch_obs.append(obs_tensor)
                batch_actions.append(actions)
                batch_logprobs.append(log_probs.detach())
                batch_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
                batch_dones.append(torch.tensor(dones, dtype=torch.float32, device=device))
                batch_values.append(values.detach())

                obs = next_obs

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            _, next_values = model(obs_tensor)

        if update == 0:
            sys.stdout.write("post-rollout: building advantages\n")
            sys.stdout.flush()
        if update == 0 and opt_progress_fn is not None:
            batch_size_est = ppo_cfg.rollout_steps * ppo_cfg.num_envs
            total_opt_steps = ppo_cfg.update_epochs * math.ceil(batch_size_est / ppo_cfg.minibatch_size)
            opt_progress_fn(0, total_opt_steps)

        advantages, returns = _compute_gae(
            batch_rewards,
            batch_dones,
            batch_values,
            next_values,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.gae_lambda,
        )

        b_obs = torch.cat(batch_obs, dim=0)
        b_actions = torch.cat(batch_actions, dim=0)
        b_logprobs = torch.cat(batch_logprobs, dim=0)
        b_adv = advantages.flatten()
        b_returns = returns.flatten()
        b_values = torch.cat(batch_values, dim=0).flatten()

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        batch_size = b_obs.shape[0]

        total_opt_steps = ppo_cfg.update_epochs * math.ceil(batch_size / ppo_cfg.minibatch_size)
        opt_step = 0
        for epoch in range(ppo_cfg.update_epochs):
            seed = ppo_cfg.seed + update * 1000 + epoch
            for idx in _batch_indices(batch_size, ppo_cfg.minibatch_size, seed):
                opt_step += 1
                if update == 0 and opt_progress_fn is not None:
                    opt_progress_fn(opt_step, total_opt_steps)
                mb_obs = b_obs[idx]
                mb_actions = b_actions[idx]
                mb_logprobs = b_logprobs[idx]
                mb_adv = b_adv[idx]
                mb_returns = b_returns[idx]

                logits, values = model(mb_obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - mb_logprobs).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * (mb_returns - values).pow(2).mean()

                loss = policy_loss + ppo_cfg.vf_coef * value_loss - ppo_cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

        avg_return = b_returns.mean().item()
        if progress_fn is not None:
            progress_fn(update + 1, ppo_cfg.total_updates, avg_return)
            if (update + 1) % max(1, ppo_cfg.total_updates // 10) == 0:
                print(f"update={update + 1} avg_return={avg_return:.4f}")
            elapsed = time.monotonic() - update_start
            print(f"update_time={elapsed:.2f}s")
    finally:
        if ppo_cfg.use_mp_envs:
            env_pool.close()

    return model


def _compute_gae(
    rewards: Iterable[torch.Tensor],
    dones: Iterable[torch.Tensor],
    values: Iterable[torch.Tensor],
    next_values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rewards_t = torch.stack(list(rewards))
    dones_t = torch.stack(list(dones))
    values_t = torch.stack(list(values))

    steps, num_envs = rewards_t.shape
    advantages = torch.zeros((steps, num_envs), device=rewards_t.device)
    last_gae = torch.zeros(num_envs, device=rewards_t.device)
    for t in reversed(range(steps)):
        next_val = next_values if t == steps - 1 else values_t[t + 1]
        mask = 1.0 - dones_t[t]
        delta = rewards_t[t] + gamma * next_val * mask - values_t[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values_t
    return advantages, returns


def _batch_indices(total: int, batch_size: int, seed: int) -> Iterable[torch.Tensor]:
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    for start in range(0, total, batch_size):
        yield torch.tensor(indices[start : start + batch_size], dtype=torch.long)
class _AsyncEnvPool:
    def __init__(self, env_cfg: LiquidationEnvConfig, num_envs: int) -> None:
        self._ctx = mp.get_context("spawn")
        self._conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []
        cfg_dict = _env_cfg_to_dict(env_cfg)
        for i in range(num_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(target=_env_worker, args=(child_conn, cfg_dict, i))
            proc.daemon = True
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
        self.obs = []
        for conn in self._conns:
            if not conn.poll(30.0):
                raise RuntimeError("env worker did not respond during init")
            msg = conn.recv()
            if msg[0] == "error":
                raise RuntimeError(f"env worker failed during init: {msg[1]}")
            if msg[0] != "obs":
                raise RuntimeError(f"unexpected env init message: {msg}")
            self.obs.append(msg[1])

    def step(self, actions: list[int]) -> tuple[list[list[float]], list[float], list[float]]:
        for conn, action in zip(self._conns, actions):
            conn.send(("step", int(action)))
        next_obs = []
        rewards = []
        dones = []
        for conn in self._conns:
            msg = conn.recv()
            if msg[0] == "error":
                raise RuntimeError(f"env worker failed during step: {msg[1]}")
            if msg[0] != "step":
                raise RuntimeError(f"unexpected env step message: {msg}")
            _, obs, reward, done = msg
            next_obs.append(obs)
            rewards.append(float(reward))
            dones.append(float(done))
        return next_obs, rewards, dones

    def close(self) -> None:
        for conn in self._conns:
            conn.send(("close", None))
        for proc in self._procs:
            proc.join(timeout=1.0)


def _env_cfg_to_dict(cfg: LiquidationEnvConfig) -> dict:
    d = asdict(cfg)
    fmt = d.pop("historical_format")
    if hasattr(fmt, "name"):
        d["historical_format_str"] = fmt.name.lower()
    else:
        d["historical_format_str"] = str(fmt).split(".")[-1].lower()
    if d.get("historical_paths") is not None:
        d["historical_paths"] = list(d["historical_paths"])
    return d


def _env_worker(conn: mp.connection.Connection, cfg_dict: dict, idx: int) -> None:
    from sim.flow import HistoricalFormat

    fmt_map = {
        "truefx": HistoricalFormat.TRUEFX,
        "csv": HistoricalFormat.CSV,
        "binary": HistoricalFormat.BINARY,
    }
    cfg_local = dict(cfg_dict)
    fmt_key = cfg_local.pop("historical_format_str", "truefx")
    fmt = fmt_map.get(fmt_key, HistoricalFormat.TRUEFX)
    cfg_local["seed"] = int(cfg_local.get("seed", 0)) + idx
    env_cfg = LiquidationEnvConfig(**cfg_local, historical_format=fmt)
    try:
        env = InventoryLiquidationEnv(env_cfg, flow_cfg=None)
        obs = env.reset()
        conn.send(("obs", obs))
        while True:
            cmd, payload = conn.recv()
            if cmd == "step":
                obs, reward, done, _info = env.step(int(payload))
                if done:
                    obs = env.reset()
                conn.send(("step", obs, reward, done))
            elif cmd == "close":
                break
    except Exception as exc:
        try:
            conn.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        conn.close()
