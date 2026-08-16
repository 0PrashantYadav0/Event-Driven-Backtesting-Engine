# Event-Driven Backtesting Engine

A production-grade, **event-driven** backtesting engine for systematic trading
strategies, with realistic transaction-cost & slippage modelling, a clean
strategy interface, and **walk-forward validation** to guard against
overfitting. Ships with two real strategies (trend-following momentum and
cointegration-based mean-reversion pairs trading) and a fully deterministic,
**offline** synthetic-data generator so it runs anywhere with zero downloads.

> Why event-driven and not vectorised? A vectorised backtest is fast to write
> but silently invites look-ahead bias and can't model order-by-order fills,
> partial state, or path-dependent costs. This engine mirrors how a live
> trading system actually works: components communicate only through a queue of
> immutable events, so the same strategy code could, in principle, be pointed
> at a live feed.

---

## Architecture

A single FIFO event queue drives four decoupled components. Each bar cascades
`MARKET → SIGNAL → ORDER → FILL` until the queue drains, then the next bar is
pulled.

```
   +-------------+   MarketEvent   +-----------+   SignalEvent   +-------------+
   | DataHandler | --------------> | Strategy  | --------------> |  Portfolio  |
   +-------------+                 +-----------+                 +-------------+
         ^                                                             |
         |                                                       OrderEvent
         | FillEvent                                                   v
         |                          +----------------------+  <--------+
         +------------------------- |  ExecutionHandler    |
              (updates the book)    | (costs + slippage)   |
                                    +----------------------+
```

| Module            | Responsibility                                                      |
|-------------------|---------------------------------------------------------------------|
| `events.py`       | Immutable, typed events (`Market/Signal/Order/Fill`).               |
| `data.py`         | `DataHandler` ABC, CSV + in-memory handlers, synthetic generators.  |
| `strategy.py`     | `Strategy` ABC, `MomentumStrategy`, `MeanReversionPairsStrategy`.   |
| `portfolio.py`    | Position sizing, order generation, cash/P&L accounting, equity curve.|
| `execution.py`    | Simulated fills with commission + adverse slippage.                 |
| `performance.py`  | Sharpe, max drawdown, CAGR, hit-rate, volatility.                   |
| `backtest.py`     | The event loop / engine.                                            |
| `walkforward.py`  | Rolling/anchored walk-forward optimisation & OOS aggregation.       |
| `main.py`         | CLI + end-to-end demo.                                              |

---

## No look-ahead bias — a structural guarantee

Look-ahead bias is not merely *discouraged*, it is **impossible by
construction**: the `DataHandler` only reveals bars as `update_bars()` advances
its cursor, so a strategy calling `get_latest_bars()` physically cannot see a
bar that hasn't happened yet. This is asserted directly in
`tests/test_strategy.py::test_strategy_never_sees_future_bars` (on call *k* the
strategy sees exactly *k* bars) and enforced at the data layer in
`tests/test_data.py::test_in_memory_handler_no_lookahead`.

Fills occur at the current bar's close *after* equity is marked, so trading
costs land in the following period — never retroactively.

> **Caveat, stated honestly:** fills happen at the same bar's close the signal
> was computed on, i.e. we assume zero-latency execution at the decision price.
> This is *not* look-ahead (the strategy only ever sees bars ≤ *t*), but a
> stricter, more conservative mode would fill at `open[t+1]`. That's on the
> roadmap (`fill_delay`).

---

## Quickstart

```bash
# create the env (uv) and install deps
uv venv
uv pip install numpy pandas pytest        # add your index flags on a locked-down network

# run the full demo (synthetic data, both strategies, walk-forward, benchmark)
python -m backtester.main demo

# or individual pieces
python -m backtester.main momentum
python -m backtester.main pairs
python -m backtester.main walkforward
python -m backtester.main bench

# run the tests
python -m pytest
```

---

## Results (deterministic, `seed=7`, 1000 daily bars)

Single-pass backtest (commission 1 bp + $1 floor, slippage 2 bps):

| Strategy                 | Total return | CAGR   | Sharpe | Sortino | Max DD  | Trades |
|--------------------------|-------------:|-------:|-------:|--------:|--------:|-------:|
| Momentum (dual-MA)       |      −1.93%  | −0.49% |  0.03  |  0.05   | −21.9%  |   34   |
| Mean-reversion pairs     |      +3.60%  | +0.90% |  0.19  |  0.29   | −43.3%  |   92   |

**Walk-forward (out-of-sample, stitched across 4 folds):**

| Strategy               | OOS total return | OOS CAGR | OOS Sharpe | OOS Max DD |
|------------------------|-----------------:|---------:|-----------:|-----------:|
| Momentum               |         +16.26%  |  +6.53%  |    0.52    |   −18.6%   |
| Mean-reversion pairs   |         −14.70%  |  −6.46%  |   −0.06    |   −40.9%   |

**Throughput:** ~135,000 events/second (single core, Python; `bench` command).

> These numbers come from *synthetic* data and are for demonstrating the
> machinery, not a claim of alpha. The honest takeaway is exactly what a
> researcher wants to see: the pairs strategy posts a strong **in-sample**
> Sharpe (~1.3–1.7 per fold) that collapses to **negative out-of-sample** —
> a textbook overfit that walk-forward validation exposes rather than hides.
> That is the entire point of building this the hard way.

---

## Design trade-offs

- **Event-driven over vectorised:** ~100–1000× slower than a NumPy one-liner,
  bought in exchange for realism, live-parity, and immunity to look-ahead.
  Still ~140k events/s, which is plenty for research.
- **Target-position sizing with a book-level leverage cap:** the portfolio
  sizes to a *target* fraction of equity, so rebalances and flattens fall out
  naturally and positions never accidentally stack. Aggregate gross exposure is
  capped at `max_gross_leverage * equity` across *all* legs on a bar (verified
  in `tests/test_review_fixes.py`).
- **Beta-weighted pairs:** the pairs strategy sizes its two legs by the OLS
  hedge ratio it estimates each bar, so the executed trade matches the spread
  being modelled (a naive equal-notional version blew up to ~55% vol; the
  beta-weighted version is ~33%).
- **Costs baked into the fill price:** slippage moves the fill adversely and
  commission is charged separately, so P&L already reflects frictions with no
  double counting.
- **Synthetic data by default:** guarantees determinism and offline runs; real
  CSVs (`timestamp,open,high,low,close,volume`) drop straight into
  `HistoricCSVDataHandler`.

---

## Roadmap

- Cointegration test (ADF / Johansen) as an entry gate for the pairs strategy.
- Next-bar-open fills (`fill_delay`) as an even more conservative execution mode.
- Limit orders + partial fills; volume-participation slippage model.
- Numba/Cython hot loop for tick-level data.

---

## Testing

62 unit/integration tests cover performance math (validated against
hand-computed values), event immutability, fill cost/slippage math, portfolio
accounting invariants (including shorts and a book-level leverage cap), strategy
signal logic, the no-look-ahead guarantee, walk-forward correctness (anchored &
rolling, OOS trade attribution), non-finite-data rejection, and an end-to-end
deterministic backtest. `RuntimeWarning`s are promoted to errors so numerical
issues can't hide.

```bash
python -m pytest -q
```
