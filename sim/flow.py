from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Literal

from lob.book import LimitOrderBook
from lob.types import Order

EventKind = Literal["LIMIT", "MARKET", "CANCEL"]

@dataclass
class FlowConfig:
    #Reference mid when book has no mid yet
    initial_mid: int = 10000
    
    #Time step per event
    dt_us: int = 1000
    
    #Event Probabilities
    p_limit: float = 0.40
    p_market: float = 0.35
    p_cancel: float = 0.25

    cancel_levels: int = 5
    soft_max_orders: int = 20000
    hard_max_orders: int = 50000
    cancel_boost: float = 0.50
    
    #Limit Placement
    min_offset: int = 1
    max_offset: int = 5

    # Bias limit placements toward the touch (smaller offsets get more weight).
    # Weight for offset k is proportional to 1 / k**offset_power.
    offset_power: float = 2.0
    
    #Quantity Range
    min_qty: int = 1
    max_qty: int = 3

    # Market order quantity range (larger so market orders can walk multiple levels)
    min_market_qty: int = 1
    max_market_qty: int = 12

    # Liquidity floor: if top-of-book is too empty, force LIMIT orders to replenish (keep small so thinning can occur).
    min_touch_qty: int = 5

    cancel_fallback_to_limit: bool = True

    def __post_init__(self) -> None:
        # Ensure base event probabilities are valid and sum to 1.
        probs = (self.p_limit, self.p_market, self.p_cancel)
        if any(p < 0.0 or p > 1.0 for p in probs):
            raise ValueError(f"Event probabilities must be in [0,1], got: {probs}")
        s = self.p_limit + self.p_market + self.p_cancel
        if abs(s - 1.0) > 1e-9:
            raise ValueError(
                f"Event probabilities must sum to 1.0, got sum={s} from (p_limit={self.p_limit}, p_market={self.p_market}, p_cancel={self.p_cancel})"
            )
    
class PoissonOrderFlow:
    def __init__(self, config: FlowConfig, seed: Optional[int] = 0):
        self.config = config
        self.rng = random.Random(seed)
        self._next_order_id = 1
        self._ref_mid = config.initial_mid
    
    def _mid_tick(self, book: LimitOrderBook) -> int:
        m = book.mid()
        if m is None:
            return self._ref_mid
        return int(round(m))
    
    def _new_order_id(self) -> int:
        order_id = self._next_order_id
        self._next_order_id += 1
        return order_id
    
    def sample_event_kind(self, p_limit: float, p_market: float, p_cancel: float) -> EventKind:
        r = self.rng.random()
        if r < p_limit:
            return "LIMIT"
        elif r < p_limit + p_market:
            return "MARKET"
        else:
            return "CANCEL"
        
    def sample_offset(self) -> int:
        lo = self.config.min_offset
        hi = self.config.max_offset
        if hi <= lo:
            return lo
        offsets = list(range(lo, hi + 1))
        p = self.config.offset_power
        # Heavier weight near the touch.
        weights = [1.0 / (k ** p) for k in offsets]
        return self.rng.choices(offsets, weights=weights, k=1)[0]
    
    def step(self, book: LimitOrderBook, ts: int) -> tuple[EventKind, int]:
        n_orders = len(book.orders)
        soft = self.config.soft_max_orders
        hard = self.config.hard_max_orders
        
        if hard <= soft:
            hard = soft + 1
            
        pressure = 0.0
        
        if n_orders > soft:
            pressure = min(1.0, (n_orders - soft) / (hard - soft))
        
        p_market = self.config.p_market
        p_cancel = min(0.99, self.config.p_cancel + self.config.cancel_boost * pressure)
        p_limit = max(0.0, 1.0 - p_market - p_cancel)
        
        kind = self.sample_event_kind(p_limit, p_market, p_cancel)

        #Keep the book alive: if one or both sides are empty OR touch liquidity is too low, force LIMIT
        if book.best_bid() is None or book.best_ask() is None:
            kind = "LIMIT"
        else:
            bid_touch = book.depth("BID", levels=1)
            ask_touch = book.depth("ASK", levels=1)
            bid_q = bid_touch[0][1] if bid_touch else 0
            ask_q = ask_touch[0][1] if ask_touch else 0
            if bid_q < self.config.min_touch_qty or ask_q < self.config.min_touch_qty:
                kind = "LIMIT"
        
        if kind == "CANCEL" and not book.orders:
            if self.config.cancel_fallback_to_limit:
                kind = "LIMIT"
            else:
                return kind, 0
        
        if kind == "LIMIT":
            mid = self._mid_tick(book)
            if book.best_bid() is None and book.best_ask() is None:
                # seed both sides over time
                side = "BID" if (ts // self.config.dt_us) % 2 == 0 else "ASK"
            elif book.best_bid() is None:
                side = "BID"
            elif book.best_ask() is None:
                side = "ASK"
            else:
                side = "BID" if self.rng.random() < 0.5 else "ASK"
            offset = self.sample_offset()
            qty = self.rng.randint(self.config.min_qty, self.config.max_qty)
            
            px = mid - offset if side == "BID" else mid + offset
            oid = self._new_order_id()

            fills = book.add_limit(
                Order(
                    id = oid,
                    side = side,
                    px = px,
                    qty = qty,
                    ts = ts,
                )
            )
            
            traded = sum(f.qty for f in fills)
            
            return kind, traded

        if kind == "MARKET":
            # If there is no liquidity on the opposite side, fall back to providing liquidity.
            if book.best_bid() is None or book.best_ask() is None:
                kind = "LIMIT"
            if kind == "LIMIT":
                # fall through by re-running the LIMIT branch
                mid = self._mid_tick(book)
                if book.best_bid() is None and book.best_ask() is None:
                    side = "BID" if (ts // self.config.dt_us) % 2 == 0 else "ASK"
                elif book.best_bid() is None:
                    side = "BID"
                elif book.best_ask() is None:
                    side = "ASK"
                else:
                    side = "BID" if self.rng.random() < 0.5 else "ASK"
                offset = self.sample_offset()
                qty = self.rng.randint(self.config.min_qty, self.config.max_qty)
                px = mid - offset if side == "BID" else mid + offset
                oid = self._new_order_id()
                fills = book.add_limit(Order(id=oid, side=side, px=px, qty=qty, ts=ts))
                traded = sum(f.qty for f in fills)
                return "LIMIT", traded
            
            # Bias market orders to consume the heavier side near the touch (reduce extreme imbalance).
            im = book.imbalance(levels=1)
            if im > 0.2:
                # more bid volume -> more market sells
                side = "ASK" if self.rng.random() < 0.75 else "BID"
            elif im < -0.2:
                # more ask volume -> more market buys
                side = "BID" if self.rng.random() < 0.75 else "ASK"
            else:
                side = "BID" if self.rng.random() < 0.5 else "ASK"
            qty = self.rng.randint(self.config.min_market_qty, self.config.max_market_qty)
            
            fills = book.add_market(
                side = side,
                qty = qty,
                ts = ts,
            )
            
            traded = sum(f.qty for f in fills)
            
            return kind, traded
        
        cancel_levels = self.config.cancel_levels
        
        if not book.orders:
            return "LIMIT", 0
        
        # Bias cancels toward the heavier side near the touch.
        im = book.imbalance(levels=1)
        if im > 0.2:
            side = "BID" if self.rng.random() < 0.75 else "ASK"
        elif im < -0.2:
            side = "ASK" if self.rng.random() < 0.75 else "BID"
        else:
            side = "BID" if self.rng.random() < 0.5 else "ASK"
        px_levels = book.depth(side, levels=cancel_levels)
        
        if not px_levels:
            other = "ASK" if side == "BID" else "BID"
            px_levels = book.depth(other, levels=cancel_levels)
            side = other
        
        if px_levels:
            weights = [max(1, qty) for _, qty in px_levels]
            px = self.rng.choices([p for p, _ in px_levels], weights=weights, k = 1)[0]
            
            queue = (book.bids if side == "BID" else book.asks).get(px)
            if queue:
                oid = self.rng.choice(list(queue))
                book.cancel(oid, ts)
                return kind, 0
        
        if book.orders:
            oid = self.rng.choice(list(book.orders.keys()))
            book.cancel(oid, ts)
        
        return kind, 0