# microexec

**microexec** is a discrete-event **limit order book (LOB) simulator** for research in **market microstructure** and **optimal execution**. It pairs a C++ matching engine (pybind11) with Python experiments, including a PPO-based RL execution agent and TrueFX-based historical replay.

## Highlights

- Deterministic matching engine with price-time (FIFO) priority
- Limit/market/cancel orders, partial fills, crossing logic
- Integer ticks and microsecond timestamps
- Historical replay (TrueFX CSV) and synthetic flows (Poisson/Hawkes)
- Execution baselines (TWAP/VWAP/AC) and RL (PPO)

## Project Structure

```
microexec/
  cpp/
    src/                 # C++ engine, flow, execution, bindings
  lob/
    book.py              # Python wrapper around C++ LimitOrderBook
    types.py             # Python re-exports of C++ types/enums
    _cpp.py              # loader for lob_cpp module
  sim/
    engine.py            # run_sim wrapper around C++ engine
    flow.py              # FlowConfig/Poisson/Hawkes/Historical wrappers
    event_sim.py         # event-driven sim loop (agent, exchange)
    execution/           # TWAP/VWAP/AC wrappers (C++ backend)
    rl/
      env.py             # liquidation env
      ppo.py             # PPO trainer
  experiments/
    run_rl_ppo.py         # train PPO on historical or synthetic flow
    run_truefx.py         # TWAP/VWAP on TrueFX
    compare_rl_twap_vwap.py  # RL vs TWAP/VWAP comparison
  tests/
    test_book.py          # unit tests for matching correctness
```

## Install

```
python3 -m pip install -r requirements.md
```

Build the C++ module:

```
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

## Quickstart

Run unit tests:

```
python3 -m pytest
```

Run a simple LOB example:

```python
from lob.book import LimitOrderBook
from lob.types import Order, Side

book = LimitOrderBook()
book.add_limit(Order(id=1, side=Side.ASK, px=101, qty=5, ts=0))
fills = book.add_market(side=Side.BID, qty=3, ts=1)
print(fills)
print(book.best_bid(), book.best_ask())
```

## Strategy Demo

### 1. Train the RL Liquidation Agent

Train a PPO agent to liquidate inventory while minimizing market impact.

```
python -m experiments.run_rl_ppo
```

### 2. Run High-Frequency Market Making

Run the Avellaneda–Stoikov agent on historical EUR/USD data.

```
python -m experiments.run_mm_as \
  --historical-path data/EURUSD-2025-07.csv \
  --risk-aversion 0.1 --kappa 0.5
```

### 3. Optimize Parameters

Run a parallelized grid search to find robust inventory skew and spread parameters.

```
python -m experiments.run_mm_as_grid --max-workers 8
```

## TrueFX Data (Historical Replay)

TrueFX CSVs are used for historical replay. The C++ HistoricalOrderFlow parses TrueFX bid/ask rows and injects BBO quotes into the book as large resting liquidity.

Example (TWAP/VWAP on a single file):

```
python3 -m experiments.run_truefx --path data/EURUSD-2025-07.csv --strategy both
```

## RL: Inventory Liquidation (PPO)

The RL environment executes **market sells** over a fixed horizon. Observations include remaining inventory fraction, time fraction, mid move vs arrival, spread, and imbalance. The reward is negative implementation shortfall with a penalty for leftover inventory.

Train PPO on the September TrueFX set (auto-selects MPS if available):

```
python3 -m experiments.run_rl_ppo
```

Override defaults:

```
python3 -m experiments.run_rl_ppo --total-updates 1000 --rollout-steps 512 --num-envs 24 --num-threads 12
```

Artifacts:

- Model: `experiments/out/ppo_liquidation.pt`
- Training curve: `experiments/out/ppo_training_curve.png`

## RL vs TWAP vs VWAP

Compare the RL agent to TWAP and VWAP on the four CSVs in `data/` (excluding `data/rl_training`):

```
python3 -m experiments.compare_rl_twap_vwap --workers 8
```

Outputs:

- `experiments/out/compare_rl_twap_vwap.csv`
- `experiments/out/compare_rl_twap_vwap.png`

## Results Discussion

The core comparison metric is **implementation shortfall** (arrival mid vs execution price). For market sells, shortfall is:

- **positive** when you sell below the arrival mid (a cost)
- **negative** when the mid moves up and you sell at better prices (a gain)

If you see shortfall near zero, check that:

- `arrival_mid` is set once at episode start (not updated each step)
- evaluation uses the fixed `arrival_mid`, not the current mid
- the environment has a valid mid before the first action (historical warmup)

Evaluation also reports:

- **completion_rate**: fraction of the target quantity executed
- **unfilled_qty**: leftover inventory (penalized in reward)
- **avg_fill_px**: average execution price across the episode

### Comparison Snapshot (July 2025 sample)

Based on `experiments/out/compare_rl_twap_vwap.csv`:

- TWAP/VWAP completed the 1000 share target on all four files (completion_rate = 1.0).
- RL nearly completed on JPY pairs (completion_rate = 0.999, unfilled_qty = 1), and fully completed on USD pairs.
- RL reduced shortfall dramatically vs TWAP across all pairs (USDCAD/EURUSD/GBPJPY/EURJPY).
- VWAP is highly path‑dependent: it produced large gains on USDCAD/GBPJPY/EURJPY, but a large loss on EURUSD.

Interpretation:

- **RL learns to trade opportunistically**: it improves on naive TWAP by timing liquidity relative to mid moves while still completing fully.
- **VWAP is more path-dependent**: it benefits when the day trends favorably but can be punitive otherwise.

### Narrative Readout

The CSV also supports a clear qualitative narrative:

- **We are selling (ASK side)**. Higher execution prices produce more negative shortfall (gains). For example, USDCAD VWAP has the highest avg fill price and the most negative shortfall, which only makes sense for sells.
- **VWAP is high‑variance**. It wins big on trend days (USDCAD/GBPJPY/EURJPY) by concentrating volume when the market is strong, and it loses big on adverse days (EURUSD) by deferring execution into a selloff.
- **RL is the robust middle**. It beats TWAP on 3/4 files while avoiding VWAP’s extreme downside. It behaves like a risk‑aware execution policy that adapts to price/flow signals without over‑committing to volume‑only scheduling.

Summary view:

- **TWAP**: baseline; stable but price‑agnostic.
- **VWAP**: can be best in rallies, worst in crashes.
- **RL**: consistently competitive; trades off patience and risk to outperform TWAP with less tail risk than VWAP.

High variance across episodes is expected with historical FX data due to non-stationarity. If the RL curve is unstable, consider lowering the PPO learning rate or increasing minibatch size.

## Market Making (Avellaneda–Stoikov)

The event-driven sim includes an inventory-aware Avellaneda–Stoikov market maker with optional latency jitter, inside-spread improvement, and imbalance-aware skew. The grid search script sweeps key parameters and ranks by per-run event Sharpe.

Run a grid search with log-normal latency jitter:

```
python3 -m experiments.run_mm_as_grid \
  --order-latency-us 2000 --order-latency-jitter-us 1000 --order-latency-jitter-dist lognormal \
  --order-latency-logn-mu 6.5 --order-latency-logn-sigma 0.6 \
  --md-latency-us 500 --md-latency-jitter-us 300 --md-latency-jitter-dist lognormal \
  --md-latency-logn-mu 5.5 --md-latency-logn-sigma 0.5 \
  --snapshot-interval-us 1000 --max-msgs-per-sec 200 --min-resting-us 5000
```

Note: `sharpe_event_mean` is computed per run from event-level MTM deltas and then averaged across seeds; `sharpe_event_adj_mean` adds an inventory penalty.

Interpretation:

- `mtm_mean` is the average terminal mark-to-market across seeds, so it captures the strategy's net PnL in the simulated regime.
- `sharpe_event_mean` summarizes the stability of per-event PnL (event-level MTM deltas) and is less sensitive to tail outcomes than raw MTM.
- Use both together: high `mtm_mean` with low `sharpe_event_mean` can indicate unstable PnL, while high values of both imply robust performance.
- In the latest grid run with log-normal latency jitter, the top configs landed around `mtm_mean` ≈ 11.9k–14.9k and `sharpe_event_mean` ≈ 13.0–15.5.
- The top configs tend to quote inside the spread (`improve_best=True`) with moderate risk aversion and tight `kappa`, which increases fills but also inventory swings.

Limitations and oddities:

- PnL is driven by the synthetic flow model and overlay traffic, so results are sensitive to `overlay_p_market`, latency, and flow parameters.
- Per-event Sharpe can still look large because it is scaled by the number of events; it is not time-annualized.
- If the final MTM differs from the last event snapshot, it can shift Sharpe; we append the terminal MTM to align deltas.
- Inventory penalties are a proxy; they do not model funding or capital constraints, so extreme inventory can still look good on raw MTM.

## Roadmap

1. Book diagnostics (depth, imbalance, spread statistics)
2. Stochastic order flow models (Poisson → Hawkes)
3. Execution strategies (TWAP, VWAP, Almgren–Chriss, POV)
4. Market impact and implementation shortfall analysis
5. Adaptive / learning-based execution

## Disclaimer

This project is for educational and research purposes only.  
It is **not** intended for live trading or investment decision-making.
