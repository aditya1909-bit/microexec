from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

from lob.book import LimitOrderBook
from sim.flow import (
    FlowConfig,
    HawkesConfig,
    HawkesOrderFlow,
    HistoricalFormat,
    HistoricalOrderFlow,
    PoissonOrderFlow,
    Side,
)


@dataclass
class LiquidationEnvConfig:
    total_qty: int = 1000
    horizon_steps: int = 50
    decision_interval_events: int = 20
    action_bins: int = 11
    warmup_events: int = 500
    penalty_per_share: float = 1.0
    seed: int = 0
    use_hawkes: bool = False
    hawkes_cfg: Optional[HawkesConfig] = None
    use_historical: bool = False
    historical_path: Optional[str] = None
    historical_paths: Optional[Sequence[str]] = None
    historical_format: HistoricalFormat = HistoricalFormat.TRUEFX
    historical_fixed_qty: int = 1_000_000
    historical_price_scale: int = 100_000
    warmup_ticks: int = 500


class InventoryLiquidationEnv:
    def __init__(self, cfg: LiquidationEnvConfig, flow_cfg: Optional[FlowConfig] = None) -> None:
        self.cfg = cfg
        self.flow_cfg = flow_cfg or FlowConfig()
        self.book: Optional[LimitOrderBook] = None
        self.flow = None
        self.ts = 0
        self.arrival_mid: Optional[float] = None
        self.remaining = 0
        self.step_idx = 0
        self._historical_done = False
        self._reset_count = 0
        self._path_offset = int(cfg.seed)
        self._episode_filled_qty = 0
        self._episode_notional = 0.0

    @property
    def obs_dim(self) -> int:
        return 5

    @property
    def n_actions(self) -> int:
        return max(2, int(self.cfg.action_bins))

    def reset(self) -> list[float]:
        self.book = LimitOrderBook()
        if self.cfg.use_historical:
            path = self._select_historical_path()
            if not path:
                raise ValueError("historical_path is required when use_historical is True")
            self.flow = HistoricalOrderFlow(
                path,
                self.cfg.historical_format,
                fixed_qty=self.cfg.historical_fixed_qty,
                price_scale=self.cfg.historical_price_scale,
            )
        else:
            self.flow = (
                HawkesOrderFlow(self.cfg.hawkes_cfg or HawkesConfig(base=self.flow_cfg), self.cfg.seed)
                if self.cfg.use_hawkes
                else PoissonOrderFlow(self.flow_cfg, self.cfg.seed)
            )
        self.ts = 0
        self.remaining = int(self.cfg.total_qty)
        self.step_idx = 0
        self._historical_done = False
        self._reset_count += 1
        self._episode_filled_qty = 0
        self._episode_notional = 0.0

        if self.cfg.use_historical:
            for _ in range(self.cfg.warmup_ticks):
                if not self._step_market():
                    break
        else:
            for _ in range(self.cfg.warmup_events):
                self._step_market()

        mid = self.book.mid()
        self.arrival_mid = float(mid) if mid is not None else None
        if self.cfg.use_historical and self.arrival_mid is None:
            # Advance until we observe a valid mid so arrival_mid is well-defined.
            for _ in range(1000):
                if not self._step_market():
                    break
                mid = self.book.mid()
                if mid is not None:
                    self.arrival_mid = float(mid)
                    break
        return self._obs()

    def _select_historical_path(self) -> Optional[str]:
        if self.cfg.historical_paths:
            paths = list(self.cfg.historical_paths)
            if not paths:
                return None
            idx = (self._path_offset + self._reset_count) % len(paths)
            return paths[idx]
        return self.cfg.historical_path

    def step(self, action: int) -> Tuple[list[float], float, bool, Dict[str, float]]:
        if self.book is None or self.flow is None:
            raise RuntimeError("Environment not reset.")

        self.step_idx += 1
        remaining_before = self.remaining

        action = int(max(0, min(self.n_actions - 1, action)))
        frac = 0.0 if self.n_actions == 1 else action / (self.n_actions - 1)
        target_qty = int(round(remaining_before * frac))
        target_qty = max(0, min(remaining_before, target_qty))

        filled_qty = 0
        avg_px = None
        if target_qty > 0:
            fills = self.book.add_market(Side.ASK, target_qty, self.ts)
            filled_qty = sum(int(f.qty) for f in fills)
            if filled_qty > 0:
                notional = sum(float(f.qty) * float(f.px) for f in fills)
                avg_px = notional / filled_qty
                self._episode_filled_qty += filled_qty
                self._episode_notional += notional
        self.remaining = max(0, remaining_before - filled_qty)

        for _ in range(self.cfg.decision_interval_events):
            if not self._step_market():
                break

        reward = 0.0
        if filled_qty > 0 and self.arrival_mid is not None and avg_px is not None:
            reward = -self._shortfall(self.arrival_mid, avg_px, filled_qty, Side.ASK)

        done = self.remaining <= 0 or self.step_idx >= self.cfg.horizon_steps or self._historical_done
        if done and self.remaining > 0:
            reward -= float(self.cfg.penalty_per_share) * self.remaining

        obs = self._obs()
        episode_avg_px = (
            self._episode_notional / self._episode_filled_qty
            if self._episode_filled_qty > 0
            else math.nan
        )
        info = {
            "step_filled_qty": float(filled_qty),
            "step_avg_px": float(avg_px) if avg_px is not None else math.nan,
            "episode_filled_qty": float(self._episode_filled_qty),
            "episode_avg_px": float(episode_avg_px),
            "remaining": float(self.remaining),
            "arrival_mid": float(self.arrival_mid) if self.arrival_mid is not None else math.nan,
            "mid": float(self._safe_mid()),
        }
        return obs, reward, done, info

    def _step_market(self) -> bool:
        if self.cfg.use_historical:
            ok, ts, _ = self.flow.step(self.book)
            if not ok:
                self._historical_done = True
                return False
            self.ts = int(ts)
            return True
        self.ts += int(self.flow_cfg.dt_us)
        self.flow.step(self.book, self.ts)
        return True

    def _safe_mid(self) -> float:
        mid = self.book.mid() if self.book is not None else None
        return float(mid) if mid is not None else 0.0

    def _obs(self) -> list[float]:
        mid = self._safe_mid()
        spread = float(self.book.spread() or 0) if self.book is not None else 0.0
        imbalance = float(self.book.imbalance(levels=1) if self.book is not None else 0.0)
        remaining_frac = 0.0 if self.cfg.total_qty <= 0 else self.remaining / self.cfg.total_qty
        time_frac = min(1.0, self.step_idx / max(1, self.cfg.horizon_steps))
        if self.arrival_mid is None or self.arrival_mid == 0.0:
            mid_move = 0.0
        else:
            mid_move = (mid - self.arrival_mid) / self.arrival_mid
        return [remaining_frac, time_frac, mid_move, spread, imbalance]

    @staticmethod
    def _shortfall(arrival_mid: float, avg_px: float, filled_qty: int, side: Side) -> float:
        if side == Side.BID:
            return (avg_px - arrival_mid) * filled_qty
        return (arrival_mid - avg_px) * filled_qty
