# microexec

**microexec** is a discrete-event **limit order book (LOB) simulator** designed for research in **market microstructure** and **optimal execution**.

The project emphasizes **correctness, price–time priority, and reproducibility** as a foundation for studying order flow, market impact, and execution strategies.

---

## Current Scope (v0)

The repository currently implements a **fully tested core matching engine** with:

- Discrete limit order book
- Price–time (FIFO) priority
- Limit, market, and cancel orders
- Partial fills and crossing logic
- Integer price ticks and integer microsecond timestamps
- Deterministic behavior with unit tests

This core is intended to be extended with stochastic order flow models and execution algorithms.

---

## Design Choices

- **Prices:** integer ticks  
- **Time:** integer microseconds  
- **Matching rule:** execute at resting order price  
- **Priority:** strict price → time (FIFO within price level)  
- **Architecture:** minimal state, explicit data structures, test-first development

These choices mirror common abstractions used in academic microstructure models and trading simulations.

---

## Project Structure

```
microexec/
  lob/
    book.py        # limit order book + matching engine
    types.py       # Order and Fill dataclasses
  tests/
    test_book.py   # unit tests for matching correctness
```

---

## Example Usage

```python
from lob.book import LimitOrderBook
from lob.types import Order

book = LimitOrderBook()

book.add_limit(Order(id=1, side="ASK", px=101, qty=5, ts=0))
fills = book.add_market(side="BID", qty=3, ts=1)

print(fills)
print(book.best_bid(), book.best_ask())
```

---

## Testing

All matching logic is covered by unit tests.

From the project root:

```bash
python -m pytest
```

---

## Roadmap

Planned next steps:

1. Book diagnostics (depth, imbalance, spread statistics)
2. Stochastic order flow models (Poisson → Hawkes)
3. Execution strategies (TWAP, POV, Almgren–Chriss)
4. Market impact and implementation shortfall analysis
5. Adaptive / learning-based execution

---

## Motivation

This project is intended as a **research-grade microstructure sandbox**, not a black-box trading system.  
The focus is on **mechanistic clarity**, **statistical rigor**, and **realistic constraints** rather than predictive claims.

---

## Disclaimer

This project is for educational and research purposes only.  
It is **not** intended for live trading or investment decision-making.
