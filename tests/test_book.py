import pytest
from lob.book import LimitOrderBook
from lob.types import Order

def test_empty_book():
    book = LimitOrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None

def test_add_limit():
    book = LimitOrderBook()
    order = Order(id=1, side="BID", px=100, qty=10, ts=0)
    fills = book.add_limit(order)
    assert fills == []
    assert book.best_bid() == 100
    assert book.best_ask() is None

def test_crossing_limit():
    book = LimitOrderBook()
    book.add_limit(Order(1, "ASK", 101, 5, 0))
    fills = book.add_limit(Order(2, "BID", 105, 5, 1))

    assert len(fills) == 1
    assert fills[0].px == 101
    assert book.best_bid() is None
    assert book.best_ask() is None


def test_partial_fill():
    book = LimitOrderBook()
    book.add_limit(Order(1, "ASK", 101, 10, 0))
    fills = book.add_market("BID", 4, 1)

    assert fills[0].qty == 4
    assert book.best_ask() == 101


def test_fifo():
    book = LimitOrderBook()
    book.add_limit(Order(1, "ASK", 101, 5, 0))
    book.add_limit(Order(2, "ASK", 101, 5, 1))

    fills = book.add_market("BID", 7, 2)

    assert fills[0].order_id == 1
    assert fills[1].order_id == 2
    assert fills[1].qty == 2


def test_cancel():
    book = LimitOrderBook()
    book.add_limit(Order(1, "BID", 99, 5, 0))
    ok = book.cancel(1, 1)

    assert ok
    assert book.best_bid() is None


def test_crossing_with_remainder():
    book = LimitOrderBook()
    book.add_limit(Order(1, "ASK", 101, 3, 0))
    fills = book.add_limit(Order(2, "BID", 105, 10, 1))

    assert sum(f.qty for f in fills) == 3
    assert book.best_bid() == 105
    
def test_depth_aggregates_quantity_by_price_level():
    book = LimitOrderBook()
    
    book.add_limit(Order(1, "ASK", 101, 3, 0))
    book.add_limit(Order(2, "ASK", 101, 7, 1))
    book.add_limit(Order(3, "ASK", 102, 5, 2))
    
    assert book.depth("ASK", levels = 2) == [(101, 10), (102, 5)]

def test_depth_orders_best_outward():
    book = LimitOrderBook()
    book.add_limit(Order(1, "BID", 99, 1, 0))
    book.add_limit(Order(2, "BID", 101, 2, 1))
    book.add_limit(Order(3, "BID", 100, 3, 2))
    
    assert book.depth("BID", levels = 2) == [(101, 2), (100, 3)]

def test_imbalance_range_and_empty_book():
    book = LimitOrderBook()
    assert book.imbalance(levels = 1) == 0.0
    
    book.add_limit(Order(1, "BID", 100, 10, 0))
    book.add_limit(Order(2, "ASK", 101, 2, 1))
    
    im = book.imbalance(levels = 1)
    assert -1.0 <= im <= 1.0
    assert im > 0.0