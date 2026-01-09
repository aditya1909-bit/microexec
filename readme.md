# microexec

**microexec** is a discrete-event **limit order book (LOB) simulator** designed for research in **market microstructure** and **optimal execution**.

The project emphasizes **correctness, price–time priority, and reproducibility** as a foundation for studying order flow, market impact, and execution strategies.

---

## Current Scope (v0)

The repository currently implements a **core matching engine** with a **C++ backend** exposed to Python via **pybind11**:

- Discrete limit order book
- Price–time (FIFO) priority
- Limit, market, and cancel orders
- Partial fills and crossing logic
- Integer price ticks and integer microsecond timestamps
- Deterministic behavior with unit tests

The Python API calls into the C++ engine for simulation and execution sweeps.

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
  cpp/
    src/           # C++ engine, flow, execution, bindings
    CMakeLists.txt # builds lob_cpp Python module
  lob/
    book.py        # Python wrapper around C++ LimitOrderBook
    types.py       # Python re-exports of C++ types/enums
    _cpp.py        # loader for lob_cpp module
  sim/
    engine.py      # run_sim wrapper around C++ engine
    flow.py        # FlowConfig/PoissonOrderFlow wrappers
    execution/     # TWAP/VWAP wrappers (C++ backend)
  tests/
    test_book.py   # unit tests for matching correctness
```

---

## Example Usage

```python
from lob.book import LimitOrderBook
from lob.types import Order, Side

book = LimitOrderBook()

book.add_limit(Order(id=1, side=Side.ASK, px=101, qty=5, ts=0))
fills = book.add_market(side=Side.BID, qty=3, ts=1)

print(fills)
print(book.best_bid(), book.best_ask())
```

---

## Testing

All matching logic is covered by unit tests (C++ backend via Python).

From the project root:

```bash
python3 -m pytest
```

---

## Build the C++ Module

Build the `lob_cpp` Python extension from the project root:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
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
