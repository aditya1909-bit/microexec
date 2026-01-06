from collections import deque
from bisect import bisect_left
from typing import Dict, Deque, List, Optional

from .types import Order, Fill

class LimitOrderBook:
    def __init__(self):
        self.bids: Dict[int, Deque[int]] = {}
        self.asks: Dict[int, Deque[int]] = {}
        
        self.bid_prices: List[int] = []
        self.ask_prices: List[int] = []
        
        self.orders: Dict[int, Order] = {}
    
    #Helpers
    def best_bid(self) -> Optional[int]:
        return self.bid_prices[-1] if self.bid_prices else None
    
    def best_ask(self) -> Optional[int]:
        return self.ask_prices[0] if self.ask_prices else None
    
    def mid(self) -> Optional[float]:
        if self.best_bid() is None or self.best_ask() is None:
            return None
        return 0.5 * (self.best_bid() + self.best_ask())
    
    def spread(self) -> Optional[int]:
        if self.best_bid() is None or self.best_ask() is None:
            return None
        return self.best_ask() - self.best_bid()
    
    def depth(self, side: str, levels: int = 5) -> List[tuple[int, int]]:
        if levels <= 0:
            return []

        if side == "BID":
            book = self.bids
            prices = self.bid_prices
            # best bid is highest price
            pxs = list(reversed(prices[-levels:]))
        elif side == "ASK":
            book = self.asks
            prices = self.ask_prices
            # best ask is lowest price
            pxs = prices[:levels]
        else:
            raise ValueError(f"Invalid side, must be BID or ASK, got: {side}")

        out: List[tuple[int, int]] = []
        for px in pxs:
            q = book.get(px)
            if not q:
                continue
            total = 0
            for oid in q:
                o = self.orders.get(oid)
                if o is not None:
                    total += o.qty
            out.append((px, total))
        return out
    
    def imbalance(self, levels: int = 1) -> float:
        bid_vol = sum(qty for _, qty in self.depth("BID", levels=levels))
        ask_vol = sum(qty for _, qty in self.depth("ASK", levels=levels))
        denom = bid_vol + ask_vol
        if denom == 0:
            return 0.0
        return (bid_vol - ask_vol) / denom
        
    
    #Core API
    
    def add_limit(self, order: Order) -> List[Fill]:
        fills: List[Fill] = []
        
        #Crossing
        if order.side == "BID":
            while order.qty > 0 and self.best_ask() is not None and order.px >= self.best_ask():
                fills.extend(self._match(order, "ASK"))
        else:
            while order.qty > 0 and self.best_bid() is not None and order.px <= self.best_bid():
                fills.extend(self._match(order, "BID"))
    
        #Remainder Rests
        if order.qty > 0:
            self._add_to_book(order)
        
        return fills
    
    def add_market(self, side: str, qty: int, ts: int) -> List[Fill]:
        dummy = Order(
            id = -1,
            side = side,
            px = 0,
            qty = qty,
            ts = ts,
            order_type = "MARKET"
        )
        
        fills: List[Fill] = []
        opp = "ASK" if side == "BID" else "BID"
        
        while dummy.qty > 0 and ((self.best_ask() if side == "BID" else self.best_bid()) is not None):
            fills.extend(self._match(dummy, opp))
        
        return fills
    
    def cancel(self, order_id: int, ts:int) -> bool:
        if order_id not in self.orders:
            return False
        
        order = self.orders.pop(order_id)
        book = self.bids if order.side == "BID" else self.asks
        prices = self.bid_prices if order.side == "BID" else self.ask_prices
        
        q = book[order.px]
        q.remove(order_id)
        
        if not q:
            del book[order.px]
            prices.remove(order.px)
            
        return True
    
    #Private
    
    def _add_to_book(self, order: Order):
        book = self.bids if order.side == "BID" else self.asks
        prices = self.bid_prices if order.side == "BID" else self.ask_prices
        
        if order.px not in book:
            book[order.px] = deque()
            idx = bisect_left(prices, order.px)
            prices.insert(idx, order.px)
            
        book[order.px].append(order.id)
        self.orders[order.id] = order
    
    def _match(self, incoming: Order, resting_side: str) -> List[Fill]:
        fills: List[Fill] = []
        
        book = self.asks if resting_side == "ASK" else self.bids
        prices = self.ask_prices if resting_side == "ASK" else self.bid_prices
        
        if not prices:
            return fills
        
        px = prices[0] if resting_side == "ASK" else prices[-1]
        queue = book[px]
        
        while incoming.qty > 0 and queue:
            order_id = queue[0]
            resting = self.orders[order_id]
            
            traded = min(incoming.qty, resting.qty)
            
            incoming.qty -= traded
            resting.qty -= traded
            
            fills.append(
                Fill(
                    order_id = resting.id,
                    px = px,
                    qty = traded,
                    ts = incoming.ts,
                    liquidity = "MAKER",
                )
            )
            
            if resting.qty == 0:
                queue.popleft()
                del self.orders[order_id]
        if not queue:
            del book[px]
            prices.remove(px)
        
        return fills