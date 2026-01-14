from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from enum import Enum, auto
import heapq
import random
from typing import Any, Deque, Dict, Optional, Sequence

from lob.book import LimitOrderBook
from lob.types import Order, OrderType, Side
from sim.flow import (
    EventKind,
    FlowConfig,
    HistoricalFormat,
    HistoricalOrderFlow,
    PoissonOrderFlow,
    HawkesConfig,
    HawkesOrderFlow,
)


class EventType(Enum):
    MARKET_EVENT = auto()
    AGENT_SEND_ORDER = auto()
    AGENT_CANCEL = auto()
    EXCHANGE_ACK = auto()
    EXCHANGE_REJECT = auto()
    EXCHANGE_FILL = auto()
    EXCHANGE_CANCEL_ACK = auto()
    EXCHANGE_CANCEL_REJECT = auto()
    TIMER_EVENT = auto()
    MARKET_DATA = auto()
    TOXIC_EVENT = auto()


class OrderState(Enum):
    PENDING_NEW = auto()
    LIVE = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    PENDING_CANCEL = auto()
    CANCELED = auto()
    REJECTED = auto()


@dataclass
class Event:
    ts: int
    kind: EventType
    payload: Dict[str, Any]


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Event]] = []
        self._seq = 0

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.ts, self._seq, event))
        self._seq += 1

    def pop(self) -> Optional[Event]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def peek(self) -> Optional[Event]:
        if not self._heap:
            return None
        return self._heap[0][2]

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class ExchangeConfig:
    order_latency_us: int = 0
    md_latency_us: int = 0
    order_latency_jitter_us: int = 0
    md_latency_jitter_us: int = 0
    order_latency_jitter_dist: str = "uniform"
    md_latency_jitter_dist: str = "uniform"
    order_latency_logn_mu: float = 0.0
    order_latency_logn_sigma: float = 0.0
    md_latency_logn_mu: float = 0.0
    md_latency_logn_sigma: float = 0.0
    snapshot_interval_us: int = 0
    max_msgs_per_sec: int = 0
    min_resting_us: int = 0
    min_qty: int = 1
    toxic_prob: float = 0.0
    toxic_delay_us: int = 0
    toxic_qty_mult: float = 1.0


@dataclass
class OrderRecord:
    order: Order
    state: OrderState
    leaves_qty: int
    ack_ts: Optional[int] = None


class Exchange:
    def __init__(
        self,
        book: LimitOrderBook,
        flow: Any,
        cfg: ExchangeConfig,
        queue: EventQueue,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.book = book
        self.flow = flow
        self.cfg = cfg
        self.queue = queue
        self.orders: Dict[int, OrderRecord] = {}
        self._msg_bucket = None
        self._msg_count = 0
        self._last_snapshot_ts = 0
        self._rng = rng or random.Random(0)

    def _rate_limited(self, owner: str, ts: int) -> bool:
        if self.cfg.max_msgs_per_sec <= 0:
            return False
        bucket = ts // 1_000_000
        if self._msg_bucket != bucket:
            self._msg_bucket = bucket
            self._msg_count = 0
        self._msg_count += 1
        return self._msg_count > self.cfg.max_msgs_per_sec

    def _push_md(self, ts: int) -> None:
        if self.cfg.snapshot_interval_us > 0:
            if ts - self._last_snapshot_ts < self.cfg.snapshot_interval_us:
                return
            self._last_snapshot_ts = ts
        depth_bid = self.book.depth(Side.BID, levels=1)
        depth_ask = self.book.depth(Side.ASK, levels=1)
        bid_qty_1 = depth_bid[0][1] if depth_bid else 0
        ask_qty_1 = depth_ask[0][1] if depth_ask else 0
        md = {
            "best_bid": self.book.best_bid(),
            "best_ask": self.book.best_ask(),
            "mid": self.book.mid(),
            "spread": self.book.spread(),
            "imbalance1": self.book.imbalance(levels=1),
            "imbalance5": self.book.imbalance(levels=5),
            "bid_qty_1": bid_qty_1,
            "ask_qty_1": ask_qty_1,
        }
        self.queue.push(
            Event(
                ts + self._md_latency(),
                EventType.MARKET_DATA,
                {"snapshot": md},
            )
        )

    def _md_latency(self) -> int:
        base = int(self.cfg.md_latency_us)
        jitter = max(0, int(self.cfg.md_latency_jitter_us))
        if jitter <= 0:
            return base
        dist = (self.cfg.md_latency_jitter_dist or "uniform").lower()
        if dist == "lognormal":
            sample = self._rng.lognormvariate(self.cfg.md_latency_logn_mu, self.cfg.md_latency_logn_sigma)
            return base + int(min(float(jitter), max(0.0, sample)))
        return base + self._rng.randint(0, jitter)

    def on_market_event(self, ts: int) -> None:
        self.flow.step(self.book, ts)
        self._push_md(ts)

    def on_historical_event(self) -> Optional[int]:
        ok, ts, _ = self.flow.step(self.book)
        if not ok:
            return None
        ts_int = int(ts)
        self._push_md(ts_int)
        return ts_int

    def on_agent_send(self, ts: int, order: Order) -> None:
        if order.qty < self.cfg.min_qty:
            self.queue.push(Event(ts, EventType.EXCHANGE_REJECT, {"order": order, "reason": "min_qty"}))
            return
        if self._rate_limited(order.owner, ts):
            self.queue.push(Event(ts, EventType.EXCHANGE_REJECT, {"order": order, "reason": "rate_limit"}))
            return

        record = OrderRecord(order=order, state=OrderState.PENDING_NEW, leaves_qty=order.qty)
        self.orders[order.id] = record
        self.queue.push(Event(ts, EventType.EXCHANGE_ACK, {"order_id": order.id}))

        if order.order_type == OrderType.MARKET:
            fills = self.book.add_market(order.side, order.qty, ts)
            filled_qty = 0
            for f in fills:
                filled_qty += f.qty
                self.queue.push(Event(ts, EventType.EXCHANGE_FILL, {"order_id": f.order_id, "fill": f}))
                if f.order_id != order.id:
                    self.queue.push(Event(ts, EventType.EXCHANGE_FILL, {"order_id": order.id, "fill": f}))
            record.leaves_qty = order.qty - filled_qty
            record.state = OrderState.FILLED if record.leaves_qty == 0 else OrderState.PARTIALLY_FILLED
            self._push_md(ts)
            self._maybe_toxic(ts, order.side, filled_qty)
            return

        fills = self.book.add_limit(order)
        filled_qty = sum(f.qty for f in fills)
        for f in fills:
            self.queue.push(Event(ts, EventType.EXCHANGE_FILL, {"order_id": f.order_id, "fill": f}))
            if f.order_id != order.id:
                self.queue.push(Event(ts, EventType.EXCHANGE_FILL, {"order_id": order.id, "fill": f}))
        record.leaves_qty = order.qty - filled_qty
        if record.leaves_qty == 0:
            record.state = OrderState.FILLED
        elif filled_qty > 0:
            record.state = OrderState.PARTIALLY_FILLED
        else:
            record.state = OrderState.LIVE
            record.ack_ts = ts
        self._push_md(ts)
        self._maybe_toxic(ts, order.side, filled_qty)

    def on_agent_cancel(self, ts: int, order_id: int) -> None:
        record = self.orders.get(order_id)
        if record is None:
            self.queue.push(Event(ts, EventType.EXCHANGE_CANCEL_REJECT, {"order_id": order_id}))
            return
        if record.state not in (OrderState.LIVE, OrderState.PARTIALLY_FILLED):
            self.queue.push(Event(ts, EventType.EXCHANGE_CANCEL_REJECT, {"order_id": order_id}))
            return
        if record.ack_ts is not None and self.cfg.min_resting_us > 0:
            if ts - record.ack_ts < self.cfg.min_resting_us:
                self.queue.push(Event(ts, EventType.EXCHANGE_CANCEL_REJECT, {"order_id": order_id}))
                return
        ok = self.book.cancel(order_id, ts)
        if ok:
            record.state = OrderState.CANCELED
            self.queue.push(Event(ts, EventType.EXCHANGE_CANCEL_ACK, {"order_id": order_id}))
            self._push_md(ts)
        else:
            self.queue.push(Event(ts, EventType.EXCHANGE_CANCEL_REJECT, {"order_id": order_id}))

    def _maybe_toxic(self, ts: int, side: Side, filled_qty: int) -> None:
        if filled_qty <= 0 or self.cfg.toxic_prob <= 0.0:
            return
        if self.cfg.toxic_prob < 1.0:
            # use flow RNG if available
            r = getattr(self.flow, "rng", None)
            if r is not None:
                if r.random() > self.cfg.toxic_prob:
                    return
        qty = max(1, int(round(filled_qty * self.cfg.toxic_qty_mult)))
        self.queue.push(
            Event(
                ts + self.cfg.toxic_delay_us,
                EventType.TOXIC_EVENT,
                {"side": side, "qty": qty},
            )
        )


class Agent:
    def on_market_data(self, ts: int, snapshot: Dict[str, Any]) -> None:
        pass

    def on_ack(self, ts: int, order_id: int) -> None:
        pass

    def on_reject(self, ts: int, order: Order, reason: str) -> None:
        pass

    def on_fill(self, ts: int, order_id: int, fill: Any) -> None:
        pass

    def on_cancel_ack(self, ts: int, order_id: int) -> None:
        pass

    def on_cancel_reject(self, ts: int, order_id: int) -> None:
        pass

    def on_timer(self, ts: int) -> Any:
        return []


class TwapAgent(Agent):
    def __init__(
        self,
        *,
        total_qty: int,
        child_interval_events: int,
        owner: str = "AGENT",
        order_id_start: int = 1_000_000,
    ) -> None:
        self.total_qty = total_qty
        self.child_interval_events = child_interval_events
        self.owner = owner
        self.filled_qty = 0
        self._next_order_id = max(1, int(order_id_start))

    def on_fill(self, ts: int, order_id: int, fill: Any) -> None:
        self.filled_qty += int(fill.qty)

    def on_timer(self, ts: int) -> list[Order]:
        remaining = self.total_qty - self.filled_qty
        if remaining <= 0:
            return []
        # simple even split heuristic (ceil)
        n_slices_left = max(1, remaining // max(1, remaining))
        qty = min(remaining, max(1, remaining // n_slices_left))
        oid = self._next_order_id
        self._next_order_id += 1
        return [
            Order(
                id=oid,
                side=Side.BID,
                px=0,
                qty=qty,
                ts=ts,
                owner=self.owner,
                order_type=OrderType.MARKET,
            )
        ]


class AvellanedaStoikovAgent(Agent):
    def __init__(
        self,
        *,
        child_interval_events: int,
        horizon_events: int,
        owner: str = "MM",
        risk_aversion: float = 0.1,
        kappa: float = 1.5,
        base_qty: int = 1,
        max_inventory: int = 50,
        sigma_window: int = 200,
        min_spread_ticks: int = 1,
        imbalance_sensitivity: float = 0.0,
        imbalance_level: int = 1,
        join_best: bool = True,
        improve_best: bool = True,
        improve_ticks: int = 1,
        cross_prob: float = 0.05,
        cross_inventory_threshold: int = 1,
        cross_ticks: int = 1,
        seed: int = 0,
        order_id_start: int = 1_000_000,
    ) -> None:
        self.child_interval_events = child_interval_events
        self.horizon_events = horizon_events
        self.owner = owner
        self.risk_aversion = risk_aversion
        self.kappa = kappa
        self.base_qty = max(1, int(base_qty))
        self.max_inventory = max(1, int(max_inventory))
        self.min_spread_ticks = max(1, int(min_spread_ticks))
        self.imbalance_sensitivity = float(imbalance_sensitivity)
        self.imbalance_level = 5 if int(imbalance_level) >= 5 else 1
        self.join_best = bool(join_best)
        self.improve_best = bool(improve_best)
        self.improve_ticks = max(1, int(improve_ticks))
        self.cross_prob = max(0.0, float(cross_prob))
        self.cross_inventory_threshold = max(0, int(cross_inventory_threshold))
        self.cross_ticks = max(1, int(cross_ticks))
        self._rng = random.Random(seed)

        self.inventory = 0
        self.cash = 0.0
        self._next_order_id = max(1, int(order_id_start))
        self._live_orders: Dict[int, Order] = {}
        self._pending_cancel: set[int] = set()
        self._last_mid: Optional[float] = None
        self._returns: Deque[float] = deque(maxlen=max(5, int(sigma_window)))
        self._step_idx = 0
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        self._imbalance: Optional[float] = None
        self.mtm_series: list[float] = []
        self.n_fills = 0
        self.filled_qty = 0
        self.n_orders = 0

    def on_market_data(self, ts: int, snapshot: Dict[str, Any]) -> None:
        mid = snapshot.get("mid")
        if mid is None:
            return
        mid_val = float(mid)
        if self._last_mid is not None and self._last_mid > 0:
            r = math.log(mid_val / self._last_mid)
            self._returns.append(r)
        self._last_mid = mid_val
        self._best_bid = snapshot.get("best_bid")
        self._best_ask = snapshot.get("best_ask")
        if self.imbalance_level >= 5 and snapshot.get("imbalance5") is not None:
            self._imbalance = float(snapshot.get("imbalance5"))
        else:
            imb = snapshot.get("imbalance1")
            self._imbalance = float(imb) if imb is not None else None
        mtm = self.cash + self.inventory * mid_val
        self.mtm_series.append(mtm)

    def on_ack(self, ts: int, order_id: int) -> None:
        pass

    def on_reject(self, ts: int, order: Order, reason: str) -> None:
        self._live_orders.pop(order.id, None)
        self._pending_cancel.discard(order.id)

    def on_fill(self, ts: int, order_id: int, fill: Any) -> None:
        order = self._live_orders.get(order_id)
        if order is None:
            return
        fill_qty = int(fill.qty)
        fill_px = float(fill.px)
        self.n_fills += 1
        self.filled_qty += fill_qty
        if order.side == Side.BID:
            self.inventory += fill_qty
            self.cash -= fill_qty * fill_px
        else:
            self.inventory -= fill_qty
            self.cash += fill_qty * fill_px
        remaining = int(order.qty) - fill_qty
        if remaining <= 0:
            self._live_orders.pop(order_id, None)
            self._pending_cancel.discard(order_id)
        else:
            self._live_orders[order_id] = Order(
                id=order.id,
                side=order.side,
                px=order.px,
                qty=remaining,
                ts=order.ts,
                owner=order.owner,
                order_type=order.order_type,
            )

    def on_cancel_ack(self, ts: int, order_id: int) -> None:
        self._live_orders.pop(order_id, None)
        self._pending_cancel.discard(order_id)

    def on_cancel_reject(self, ts: int, order_id: int) -> None:
        self._pending_cancel.discard(order_id)

    def _sigma2(self) -> float:
        n = len(self._returns)
        if n < 2:
            return 0.0
        mean = sum(self._returns) / n
        var = sum((x - mean) ** 2 for x in self._returns) / max(1, n - 1)
        return float(var)

    def _reservation_and_spread(self) -> tuple[float, float]:
        if self._last_mid is None:
            return 0.0, float(self.min_spread_ticks)
        sigma2 = self._sigma2()
        remaining_steps = max(1, (self.horizon_events // max(1, self.child_interval_events)) - self._step_idx)
        t_remaining = float(remaining_steps)
        gamma = max(1e-8, float(self.risk_aversion))
        kappa = max(1e-8, float(self.kappa))
        reservation = self._last_mid - self.inventory * gamma * sigma2 * t_remaining
        if self._imbalance is not None and self.imbalance_sensitivity != 0.0:
            if self._best_bid is not None and self._best_ask is not None:
                spread_ticks = max(1.0, float(self._best_ask) - float(self._best_bid))
            else:
                spread_ticks = 1.0
            reservation += self.imbalance_sensitivity * self._imbalance * spread_ticks
        spread = (
            gamma * sigma2 * t_remaining
            + (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        )
        spread = max(spread, float(self.min_spread_ticks))
        return reservation, spread

    def _quote_qtys(self) -> tuple[int, int]:
        inv_frac = max(-1.0, min(1.0, self.inventory / max(1.0, float(self.max_inventory))))
        bid_scale = max(0.1, 1.0 - inv_frac)
        ask_scale = max(0.1, 1.0 + inv_frac)
        bid_qty = max(1, int(round(self.base_qty * bid_scale)))
        ask_qty = max(1, int(round(self.base_qty * ask_scale)))
        return bid_qty, ask_qty

    def on_timer(self, ts: int) -> list[Order]:
        self._step_idx += 1
        orders: list[Order] = []
        cancel_ids = [oid for oid in self._live_orders.keys() if oid not in self._pending_cancel]
        self._pending_cancel.update(cancel_ids)

        reservation, spread = self._reservation_and_spread()
        if reservation <= 0:
            return orders, cancel_ids

        bid_px = int(round(reservation - 0.5 * spread))
        ask_px = int(round(reservation + 0.5 * spread))
        if bid_px >= ask_px:
            ask_px = bid_px + self.min_spread_ticks

        best_bid = int(self._best_bid) if self._best_bid is not None else None
        best_ask = int(self._best_ask) if self._best_ask is not None else None
        if best_bid is not None and best_ask is not None:
            if self.improve_best and best_ask - best_bid > 1:
                bid_px = max(bid_px, best_bid + self.improve_ticks)
                ask_px = min(ask_px, best_ask - self.improve_ticks)
            elif self.join_best:
                bid_px = max(bid_px, best_bid)
                ask_px = min(ask_px, best_ask)
        elif best_ask is not None:
            bid_px = min(bid_px, best_ask - 1)
        elif best_bid is not None:
            ask_px = max(ask_px, best_bid + 1)

        cross_side = None
        if (
            self.cross_prob > 0.0
            and abs(self.inventory) <= self.cross_inventory_threshold
            and best_bid is not None
            and best_ask is not None
            and self._rng.random() < self.cross_prob
        ):
            if self.inventory > 0:
                cross_side = "ask"
            elif self.inventory < 0:
                cross_side = "bid"
            else:
                cross_side = "bid" if self._rng.random() < 0.5 else "ask"

        place_bid = True
        place_ask = True
        bid_order_type = OrderType.LIMIT
        ask_order_type = OrderType.LIMIT
        if cross_side == "bid" and best_ask is not None:
            bid_px = max(bid_px, best_ask + self.cross_ticks)
            place_ask = False
        elif cross_side == "ask" and best_bid is not None:
            ask_px = min(ask_px, best_bid - self.cross_ticks)
            place_bid = False
        else:
            if bid_px >= ask_px:
                ask_px = bid_px + self.min_spread_ticks

        if bid_px <= 0 or ask_px <= 0:
            return orders, cancel_ids

        bid_qty, ask_qty = self._quote_qtys()
        if place_bid:
            bid_id = self._next_order_id
            self._next_order_id += 1
            bid_order = Order(
                id=bid_id,
                side=Side.BID,
                px=bid_px,
                qty=bid_qty,
                ts=ts,
                owner=self.owner,
                order_type=bid_order_type,
            )
            self._live_orders[bid_id] = bid_order
            orders.append(bid_order)
        if place_ask:
            ask_id = self._next_order_id
            self._next_order_id += 1
            ask_order = Order(
                id=ask_id,
                side=Side.ASK,
                px=ask_px,
                qty=ask_qty,
                ts=ts,
                owner=self.owner,
                order_type=ask_order_type,
            )
            self._live_orders[ask_id] = ask_order
            orders.append(ask_order)
        self.n_orders += len(orders)
        return orders, cancel_ids

def run_event_sim(
    *,
    horizon_events: int,
    cfg: FlowConfig,
    exchange_cfg: ExchangeConfig,
    agent: Agent,
    seed: int = 0,
    use_hawkes: bool = False,
    hawkes_cfg: Optional[HawkesConfig] = None,
    use_historical: bool = False,
    historical_path: Optional[str] = None,
    historical_paths: Optional[Sequence[str]] = None,
    historical_format: HistoricalFormat = HistoricalFormat.TRUEFX,
    historical_fixed_qty: int = 1_000_000,
    historical_price_scale: int = 100_000,
    warmup_ticks: int = 500,
    historical_overlay_p_market: float = 0.0,
    historical_overlay_min_qty: int = 1,
    historical_overlay_max_qty: int = 5,
    historical_overlay_bid_prob: float = 0.5,
    overlay_delay_us: int = 1,
    timer_delay_us: int = 1,
    debug_first_events: int = 0,
) -> tuple[LimitOrderBook, Agent]:
    book = LimitOrderBook()
    queue = EventQueue()

    def _select_historical_path() -> Optional[str]:
        if historical_paths:
            paths = list(historical_paths)
            if not paths:
                return None
            idx = int(seed) % len(paths)
            return paths[idx]
        return historical_path

    if use_historical:
        path = _select_historical_path()
        if not path:
            raise ValueError("historical_path is required when use_historical is True")
        flow = HistoricalOrderFlow(
            path,
            historical_format,
            fixed_qty=historical_fixed_qty,
            price_scale=historical_price_scale,
        )
    else:
        flow = (
            HawkesOrderFlow(hawkes_cfg or HawkesConfig(base=cfg), seed)
            if use_hawkes
            else PoissonOrderFlow(cfg, seed)
        )
    order_rng = random.Random(int(seed) + 991)
    exchange = Exchange(book, flow, exchange_cfg, queue, rng=random.Random(int(seed) + 1337))
    overlay_rng = random.Random(seed)
    next_overlay_id = 1_000_000_000

    min_overlay_delay = max(1, int(timer_delay_us) + int(exchange_cfg.order_latency_us) + 1)

    def _order_latency() -> int:
        base = int(exchange_cfg.order_latency_us)
        jitter = max(0, int(exchange_cfg.order_latency_jitter_us))
        if jitter <= 0:
            return base
        dist = (exchange_cfg.order_latency_jitter_dist or "uniform").lower()
        if dist == "lognormal":
            sample = order_rng.lognormvariate(
                exchange_cfg.order_latency_logn_mu,
                exchange_cfg.order_latency_logn_sigma,
            )
            return base + int(min(float(jitter), max(0.0, sample)))
        return base + order_rng.randint(0, jitter)

    debug_remaining = max(0, int(debug_first_events))

    def _debug(msg: str) -> None:
        nonlocal debug_remaining
        if debug_remaining <= 0:
            return
        print(msg)
        debug_remaining -= 1

    def _maybe_overlay_market(ts_event: int) -> None:
        nonlocal next_overlay_id
        if historical_overlay_p_market <= 0.0:
            return
        if overlay_rng.random() > historical_overlay_p_market:
            return
        side = Side.BID if overlay_rng.random() < historical_overlay_bid_prob else Side.ASK
        qty = overlay_rng.randint(
            max(1, int(historical_overlay_min_qty)),
            max(1, int(historical_overlay_max_qty)),
        )
        delay = max(int(overlay_delay_us), min_overlay_delay)
        order = Order(
            id=next_overlay_id,
            side=side,
            px=0,
            qty=qty,
            ts=ts_event + delay,
            owner="FLOW",
            order_type=OrderType.MARKET,
        )
        next_overlay_id += 1
        queue.push(Event(ts_event + delay + _order_latency(), EventType.AGENT_SEND_ORDER, {"order": order}))
        _debug(f"[overlay] ts={ts_event + delay} side={order.side} qty={order.qty} delay={delay}")

    def _handle_timer_event(timer_ts: int) -> None:
        _debug(f"[timer] ts={timer_ts} inv={getattr(agent, 'inventory', None)} mid={book.mid()}")
        orders_or_actions = agent.on_timer(timer_ts)
        if isinstance(orders_or_actions, tuple):
            orders, cancel_ids = orders_or_actions
        else:
            orders = orders_or_actions
            cancel_ids = []
        for oid in cancel_ids:
            submit_ts = timer_ts + _order_latency()
            queue.push(Event(submit_ts, EventType.AGENT_CANCEL, {"order_id": oid}))
        for order in orders:
            submit_ts = timer_ts + _order_latency()
            queue.push(Event(submit_ts, EventType.AGENT_SEND_ORDER, {"order": order}))
            _debug(
                f"[quote] ts={submit_ts} id={order.id} side={order.side} px={order.px} "
                f"qty={order.qty} type={order.order_type}"
            )

    ts = 0
    # warmup
    if use_historical:
        for _ in range(warmup_ticks):
            ts_next = exchange.on_historical_event()
            if ts_next is None:
                break
            ts = ts_next
    else:
        for _ in range(500):
            ts += cfg.dt_us
            exchange.on_market_event(ts)

    if use_historical:
        events_seen = 0
        while events_seen < horizon_events:
            ts_next = exchange.on_historical_event()
            if ts_next is None:
                break
            ts = ts_next
            events_seen += 1
            _maybe_overlay_market(ts)
            if events_seen % max(1, agent.child_interval_events) == 0:
                queue.push(Event(ts + timer_delay_us, EventType.TIMER_EVENT, {}))

            while True:
                event = queue.peek()
                if event is None or event.ts > ts:
                    break
                event = queue.pop()
                if event is None:
                    break
                if event.kind == EventType.TIMER_EVENT:
                    _handle_timer_event(event.ts)
                elif event.kind == EventType.AGENT_SEND_ORDER:
                    exchange.on_agent_send(event.ts, event.payload["order"])
                elif event.kind == EventType.AGENT_CANCEL:
                    exchange.on_agent_cancel(event.ts, event.payload["order_id"])
                elif event.kind == EventType.EXCHANGE_ACK:
                    agent.on_ack(event.ts, event.payload["order_id"])
                elif event.kind == EventType.EXCHANGE_REJECT:
                    agent.on_reject(event.ts, event.payload["order"], event.payload["reason"])
                elif event.kind == EventType.EXCHANGE_FILL:
                    agent.on_fill(event.ts, event.payload["order_id"], event.payload["fill"])
                elif event.kind == EventType.EXCHANGE_CANCEL_ACK:
                    agent.on_cancel_ack(event.ts, event.payload["order_id"])
                elif event.kind == EventType.EXCHANGE_CANCEL_REJECT:
                    agent.on_cancel_reject(event.ts, event.payload["order_id"])
                elif event.kind == EventType.MARKET_DATA:
                    agent.on_market_data(event.ts, event.payload["snapshot"])
                elif event.kind == EventType.TOXIC_EVENT:
                    side = event.payload["side"]
                    qty = event.payload["qty"]
                    exchange.book.add_market(side, qty, event.ts)
                    exchange._push_md(event.ts)
    else:
        # schedule market events and timers
        for i in range(horizon_events):
            ts = (i + 1) * cfg.dt_us
            queue.push(Event(ts, EventType.MARKET_EVENT, {}))
            if i % max(1, agent.child_interval_events) == 0:
                queue.push(Event(ts + timer_delay_us, EventType.TIMER_EVENT, {}))

        while len(queue) > 0:
            event = queue.pop()
            if event is None:
                break
            ts = event.ts

            if event.kind == EventType.MARKET_EVENT:
                exchange.on_market_event(ts)
                _debug(
                    f"[market] ts={ts} best_bid={book.best_bid()} best_ask={book.best_ask()} mid={book.mid()}"
                )
                _maybe_overlay_market(ts)
            elif event.kind == EventType.TIMER_EVENT:
                _handle_timer_event(ts)
            elif event.kind == EventType.AGENT_SEND_ORDER:
                exchange.on_agent_send(ts, event.payload["order"])
            elif event.kind == EventType.AGENT_CANCEL:
                exchange.on_agent_cancel(ts, event.payload["order_id"])
            elif event.kind == EventType.EXCHANGE_ACK:
                agent.on_ack(ts, event.payload["order_id"])
            elif event.kind == EventType.EXCHANGE_REJECT:
                agent.on_reject(ts, event.payload["order"], event.payload["reason"])
            elif event.kind == EventType.EXCHANGE_FILL:
                agent.on_fill(ts, event.payload["order_id"], event.payload["fill"])
            elif event.kind == EventType.EXCHANGE_CANCEL_ACK:
                agent.on_cancel_ack(ts, event.payload["order_id"])
            elif event.kind == EventType.EXCHANGE_CANCEL_REJECT:
                agent.on_cancel_reject(ts, event.payload["order_id"])
            elif event.kind == EventType.MARKET_DATA:
                agent.on_market_data(ts, event.payload["snapshot"])
            elif event.kind == EventType.TOXIC_EVENT:
                side = event.payload["side"]
                qty = event.payload["qty"]
                exchange.book.add_market(side, qty, ts)
                exchange._push_md(ts)

    return book, agent
