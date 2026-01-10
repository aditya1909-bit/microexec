from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import heapq
from typing import Any, Dict, Optional

from lob.book import LimitOrderBook
from lob.types import Order, OrderType, Side
from sim.flow import EventKind, FlowConfig, PoissonOrderFlow, HawkesConfig, HawkesOrderFlow


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

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class ExchangeConfig:
    order_latency_us: int = 0
    md_latency_us: int = 0
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
    ) -> None:
        self.book = book
        self.flow = flow
        self.cfg = cfg
        self.queue = queue
        self.orders: Dict[int, OrderRecord] = {}
        self._msg_bucket = None
        self._msg_count = 0
        self._last_snapshot_ts = 0

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
        md = {
            "best_bid": self.book.best_bid(),
            "best_ask": self.book.best_ask(),
            "mid": self.book.mid(),
            "spread": self.book.spread(),
        }
        self.queue.push(
            Event(
                ts + self.cfg.md_latency_us,
                EventType.MARKET_DATA,
                {"snapshot": md},
            )
        )

    def on_market_event(self, ts: int) -> None:
        kind, _ = self.flow.step(self.book, ts)
        if kind in (EventKind.LIMIT, EventKind.MARKET, EventKind.CANCEL):
            self._push_md(ts)

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
                self.queue.push(Event(ts, EventType.EXCHANGE_FILL, {"order_id": order.id, "fill": f}))
            record.leaves_qty = order.qty - filled_qty
            record.state = OrderState.FILLED if record.leaves_qty == 0 else OrderState.PARTIALLY_FILLED
            self._push_md(ts)
            self._maybe_toxic(ts, order.side, filled_qty)
            return

        fills = self.book.add_limit(order)
        filled_qty = sum(f.qty for f in fills)
        for f in fills:
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

    def on_timer(self, ts: int) -> list[Order]:
        return []


class TwapAgent(Agent):
    def __init__(self, *, total_qty: int, child_interval_events: int, owner: str = "AGENT") -> None:
        self.total_qty = total_qty
        self.child_interval_events = child_interval_events
        self.owner = owner
        self.filled_qty = 0
        self._next_order_id = 1

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


def run_event_sim(
    *,
    horizon_events: int,
    cfg: FlowConfig,
    exchange_cfg: ExchangeConfig,
    agent: Agent,
    seed: int = 0,
    use_hawkes: bool = False,
    hawkes_cfg: Optional[HawkesConfig] = None,
) -> tuple[LimitOrderBook, Agent]:
    book = LimitOrderBook()
    queue = EventQueue()
    flow = (
        HawkesOrderFlow(hawkes_cfg or HawkesConfig(base=cfg), seed)
        if use_hawkes
        else PoissonOrderFlow(cfg, seed)
    )
    exchange = Exchange(book, flow, exchange_cfg, queue)

    ts = 0
    # warmup
    for _ in range(500):
        ts += cfg.dt_us
        exchange.on_market_event(ts)

    # schedule market events and timers
    for i in range(horizon_events):
        ts = (i + 1) * cfg.dt_us
        queue.push(Event(ts, EventType.MARKET_EVENT, {}))
        if i % max(1, agent.child_interval_events) == 0:
            queue.push(Event(ts, EventType.TIMER_EVENT, {}))

    while len(queue) > 0:
        event = queue.pop()
        if event is None:
            break
        ts = event.ts

        if event.kind == EventType.MARKET_EVENT:
            exchange.on_market_event(ts)
        elif event.kind == EventType.TIMER_EVENT:
            orders = agent.on_timer(ts)
            for order in orders:
                submit_ts = ts + exchange_cfg.order_latency_us
                queue.push(Event(submit_ts, EventType.AGENT_SEND_ORDER, {"order": order}))
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
