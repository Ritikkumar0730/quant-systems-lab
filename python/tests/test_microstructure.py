"""Microstructure: the C++ matching engine (through ctypes) against an
independent pure-Python book, and the Avellaneda-Stoikov experiment."""
from __future__ import annotations

import numpy as np
import pytest

from qsl.microstructure import (MarketMakingParams, OrderBook, ReferenceOrderBook, Reject, Side, TimeInForce,
                                optimal_spread, simulate)


@pytest.mark.parametrize("seed,max_orders", [(1, 4096), (2, 4096), (3, 6)])
def test_cpp_engine_matches_python_reference(seed, max_orders):
    rng = np.random.default_rng(seed)
    fast, ref = OrderBook(1, 200, max_orders), ReferenceOrderBook(1, 200, max_orders)
    ids, next_id, mid = [], 1, 100
    for step in range(20_000):
        mid = int(np.clip(mid + rng.integers(-1, 2) * (rng.random() < 0.1), 5, 195))
        r = rng.random()
        if r < 0.5:
            side = Side(int(rng.integers(2)))
            offset = int(rng.integers(-3, 9))
            price = mid - offset if side == Side.BUY else mid + offset
            if rng.random() < 0.01:
                price = 250
            qty = int(rng.integers(0, 40))
            tif = TimeInForce(int(rng.choice(4, p=[0.7, 0.1, 0.1, 0.1])))
            oid = int(rng.choice(ids)) if ids and rng.random() < 0.03 else next_id
            next_id += oid == next_id
            ids.append(oid)
            assert fast.add(oid, side, price, qty, tif) == ref.add(oid, side, price, qty, tif)
        elif r < 0.75 and ids:
            oid = int(rng.choice(ids[-300:]))
            assert fast.cancel(oid) == ref.cancel(oid)
        elif r < 0.85 and ids:
            oid, qty = int(rng.choice(ids[-300:])), int(rng.integers(0, 40))
            assert fast.reduce(oid, qty) == ref.reduce(oid, qty)
        elif r < 0.95 and ids:
            oid, price, qty = int(rng.choice(ids[-300:])), mid + int(rng.integers(-4, 5)), int(rng.integers(0, 40))
            assert fast.replace(oid, price, qty) == ref.replace(oid, price, qty)
        else:
            side, qty = Side(int(rng.integers(2))), int(rng.integers(1, 120))
            assert fast.market(next_id, side, qty) == ref.market(next_id, side, qty)
            next_id += 1
        assert fast.trades() == ref.trades()
        if step % 100 == 0:
            assert fast.depth(Side.BUY) == ref.depth(Side.BUY)
            assert fast.depth(Side.SELL) == ref.depth(Side.SELL)
            assert len(fast) == len(ref)
    assert fast.check_invariants()


def test_order_book_bridge_basics():
    book = OrderBook(1, 1000, 100)
    assert book.add(1, Side.SELL, 101, 5).reject == Reject.NONE
    assert book.add(2, Side.BUY, 100, 5).resting == 5
    assert (book.best_bid(), book.best_ask()) == (100, 101)
    ex = book.add(3, Side.BUY, 101, 8, TimeInForce.IOC)
    assert (ex.filled, ex.resting) == (5, 0)
    assert book.trades()[0].maker == 1
    assert book.add(4, Side.BUY, 100, 1, TimeInForce.POST_ONLY).reject == Reject.NONE
    with pytest.raises(ValueError):
        book.add(5, Side.BUY, 100, 1, 9)  # invalid TIF is rejected at the C boundary
    with pytest.raises(ValueError):
        OrderBook(10, 5)


def test_avellaneda_stoikov_inventory_skew_reduces_risk():
    p = MarketMakingParams()
    inv = simulate(p, "inventory", n_paths=2000, seed=1).summary()
    sym = simulate(p, "symmetric", n_paths=2000, seed=2).summary()
    # Paper (gamma = 0.1): inventory strategy P&L std ~6 vs ~13 symmetric,
    # inventory std ~3 vs ~9, at a slightly lower mean profit (~63 vs ~67).
    assert inv["std_pnl"] < 0.6 * sym["std_pnl"]
    assert inv["std_inventory"] < 0.5 * sym["std_inventory"]
    assert 55 < inv["mean_pnl"] < 75 and 55 < sym["mean_pnl"] < 80
    assert inv["sharpe_like"] > sym["sharpe_like"]
    # Terminal spread equals the pure liquidity term (2/gamma) ln(1 + gamma/k).
    assert optimal_spread(p, 0.0) == pytest.approx(20 * np.log(1 + 0.1 / 1.5), rel=1e-14)
