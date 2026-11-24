"""Python handle on the C++ matching engine, plus a pure-Python reference.

``OrderBook`` drives qsl::lob::OrderBook through the C ABI, so research code
can replay order flow against the production engine. ``ReferenceOrderBook``
is a third, independent implementation of the same contract used by the
tests to check the C++ engine and the ctypes bridge end to end.
"""
from __future__ import annotations

import ctypes
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from .._native import CppTrade, cpplib


class Side(IntEnum):
    BUY = 0
    SELL = 1


class TimeInForce(IntEnum):
    GTC = 0
    IOC = 1
    FOK = 2
    POST_ONLY = 3


class Reject(IntEnum):
    NONE = 0
    DUPLICATE_ORDER_ID = 1
    INVALID_PRICE = 2
    INVALID_QUANTITY = 3
    UNKNOWN_ORDER = 4
    BOOK_FULL = 5
    POST_ONLY_WOULD_CROSS = 6
    FILL_OR_KILL_UNFILLED = 7


@dataclass(frozen=True)
class Trade:
    taker: int
    maker: int
    price: int
    qty: int
    taker_side: Side


@dataclass(frozen=True)
class Execution:
    reject: Reject
    filled: int = 0
    resting: int = 0


def _reject(code: int) -> Reject:
    if code == 255:
        raise ValueError("invalid side or time-in-force")
    return Reject(code)


class OrderBook:
    """Integer-tick limit order book backed by the C++ engine."""

    def __init__(self, min_price: int = 1, max_price: int = 65_536, max_orders: int = 1 << 20):
        self._lib = cpplib()
        self._h = self._lib.qsl_cpp_book_new(min_price, max_price, max_orders)
        if not self._h:
            raise ValueError("invalid order book configuration")
        self.max_orders = max_orders

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.qsl_cpp_book_free(self._h)
            self._h = None

    def add(self, order_id: int, side: Side, price: int, qty: int, tif: TimeInForce = TimeInForce.GTC) -> Execution:
        filled, resting = ctypes.c_uint64(), ctypes.c_uint64()
        code = self._lib.qsl_cpp_book_add(self._h, order_id, int(side), price, qty, int(tif),
                                          ctypes.byref(filled), ctypes.byref(resting))
        return Execution(_reject(code), filled.value, resting.value)

    def market(self, order_id: int, side: Side, qty: int) -> Execution:
        filled = ctypes.c_uint64()
        code = self._lib.qsl_cpp_book_market(self._h, order_id, int(side), qty, ctypes.byref(filled))
        return Execution(_reject(code), filled.value, 0)

    def cancel(self, order_id: int) -> Reject:
        return Reject(self._lib.qsl_cpp_book_cancel(self._h, order_id))

    def reduce(self, order_id: int, new_qty: int) -> Reject:
        return Reject(self._lib.qsl_cpp_book_reduce(self._h, order_id, new_qty))

    def replace(self, order_id: int, price: int, qty: int) -> Execution:
        filled, resting = ctypes.c_uint64(), ctypes.c_uint64()
        code = self._lib.qsl_cpp_book_replace(self._h, order_id, price, qty, ctypes.byref(filled), ctypes.byref(resting))
        return Execution(Reject(code), filled.value, resting.value)

    def best(self, side: Side) -> int | None:
        price = ctypes.c_int64()
        return price.value if self._lib.qsl_cpp_book_best(self._h, int(side), ctypes.byref(price)) else None

    def best_bid(self) -> int | None:
        return self.best(Side.BUY)

    def best_ask(self) -> int | None:
        return self.best(Side.SELL)

    def volume_at(self, side: Side, price: int) -> int:
        return self._lib.qsl_cpp_book_volume_at(self._h, int(side), price)

    def __len__(self) -> int:
        return self._lib.qsl_cpp_book_order_count(self._h)

    def depth(self, side: Side, levels: int | None = None) -> list[tuple[int, int, int]]:
        """[(price, quantity, order count)], best level first."""
        cap = levels or self.max_orders
        prices = (ctypes.c_int64 * cap)()
        qtys = (ctypes.c_uint64 * cap)()
        counts = (ctypes.c_uint32 * cap)()
        n = self._lib.qsl_cpp_book_depth(self._h, int(side), prices, qtys, counts, cap)
        return [(prices[i], qtys[i], counts[i]) for i in range(n)]

    def trades(self) -> list[Trade]:
        """Executions since the last call."""
        buf = (CppTrade * 1024)()
        out: list[Trade] = []
        while n := self._lib.qsl_cpp_book_drain_trades(self._h, buf, 1024):
            out.extend(Trade(t.taker, t.maker, t.price, t.qty, Side(t.taker_side)) for t in buf[:n])
        return out

    def check_invariants(self) -> bool:
        return bool(self._lib.qsl_cpp_book_check(self._h))


class ReferenceOrderBook:
    """Slow, obviously-correct price-time priority book (dict of deques)."""

    def __init__(self, min_price: int = 1, max_price: int = 65_536, max_orders: int = 1 << 20):
        self.min_price, self.max_price, self.max_orders = min_price, max_price, max_orders
        self._levels: tuple[dict[int, deque], dict[int, deque]] = ({}, {})
        self._live: dict[int, tuple[Side, int]] = {}
        self._trades: list[Trade] = []

    def _passive_prices(self, taker: Side) -> list[int]:
        book = self._levels[1 - taker]
        return sorted(book) if taker == Side.BUY else sorted(book, reverse=True)

    def _crosses(self, side: Side, price: int, level: int) -> bool:
        return level <= price if side == Side.BUY else level >= price

    def _match(self, taker_id: int, side: Side, limit: int, qty: int) -> int:
        remaining = qty
        book = self._levels[1 - side]
        for level in self._passive_prices(side):
            if remaining == 0 or not self._crosses(side, limit, level):
                break
            queue = book[level]
            while remaining and queue:
                maker = queue[0]
                fill = min(remaining, maker[1])
                self._trades.append(Trade(taker_id, maker[0], level, fill, side))
                maker[1] -= fill
                remaining -= fill
                if maker[1] == 0:
                    queue.popleft()
                    del self._live[maker[0]]
            if not queue:
                del book[level]
        return qty - remaining

    def add(self, order_id: int, side: Side, price: int, qty: int, tif: TimeInForce = TimeInForce.GTC) -> Execution:
        if qty <= 0:
            return Execution(Reject.INVALID_QUANTITY)
        if not self.min_price <= price <= self.max_price:
            return Execution(Reject.INVALID_PRICE)
        if order_id in self._live:
            return Execution(Reject.DUPLICATE_ORDER_ID)
        full = len(self._live) >= self.max_orders
        if tif == TimeInForce.GTC and full:
            return Execution(Reject.BOOK_FULL)
        if tif == TimeInForce.POST_ONLY:
            opp = self._passive_prices(side)
            if opp and self._crosses(side, price, opp[0]):
                return Execution(Reject.POST_ONLY_WOULD_CROSS)
            if full:
                return Execution(Reject.BOOK_FULL)
        if tif == TimeInForce.FOK:
            book = self._levels[1 - side]
            available = sum(o[1] for lvl, q in book.items() if self._crosses(side, price, lvl) for o in q)
            if available < qty:
                return Execution(Reject.FILL_OR_KILL_UNFILLED)
        filled = self._match(order_id, side, price, qty)
        left = qty - filled
        if left and tif in (TimeInForce.GTC, TimeInForce.POST_ONLY):
            self._levels[side].setdefault(price, deque()).append([order_id, left])
            self._live[order_id] = (side, price)
            return Execution(Reject.NONE, filled, left)
        return Execution(Reject.NONE, filled, 0)

    def market(self, order_id: int, side: Side, qty: int) -> Execution:
        if qty <= 0:
            return Execution(Reject.INVALID_QUANTITY)
        if order_id in self._live:
            return Execution(Reject.DUPLICATE_ORDER_ID)
        limit = self.max_price if side == Side.BUY else self.min_price
        return Execution(Reject.NONE, self._match(order_id, side, limit, qty), 0)

    def _find(self, order_id: int):
        side, price = self._live[order_id]
        queue = self._levels[side][price]
        return side, price, queue, next(o for o in queue if o[0] == order_id)

    def cancel(self, order_id: int) -> Reject:
        if order_id not in self._live:
            return Reject.UNKNOWN_ORDER
        side, price, queue, order = self._find(order_id)
        queue.remove(order)
        if not queue:
            del self._levels[side][price]
        del self._live[order_id]
        return Reject.NONE

    def reduce(self, order_id: int, new_qty: int) -> Reject:
        if order_id not in self._live:
            return Reject.UNKNOWN_ORDER
        if new_qty == 0:
            return self.cancel(order_id)
        order = self._find(order_id)[3]
        if new_qty >= order[1]:
            return Reject.INVALID_QUANTITY
        order[1] = new_qty
        return Reject.NONE

    def replace(self, order_id: int, price: int, qty: int) -> Execution:
        if order_id not in self._live:
            return Execution(Reject.UNKNOWN_ORDER)
        if qty <= 0:
            return Execution(Reject.INVALID_QUANTITY)
        if not self.min_price <= price <= self.max_price:
            return Execution(Reject.INVALID_PRICE)
        side, old_price, _, order = self._find(order_id)
        if price == old_price and qty <= order[1]:
            order[1] = qty
            return Execution(Reject.NONE, 0, qty)
        self.cancel(order_id)
        return self.add(order_id, side, price, qty)

    def best_bid(self) -> int | None:
        return max(self._levels[Side.BUY], default=None)

    def best_ask(self) -> int | None:
        return min(self._levels[Side.SELL], default=None)

    def depth(self, side: Side, levels: int | None = None) -> list[tuple[int, int, int]]:
        book = self._levels[side]
        prices = sorted(book, reverse=side == Side.BUY)[:levels]
        return [(p, sum(o[1] for o in book[p]), len(book[p])) for p in prices]

    def __len__(self) -> int:
        return len(self._live)

    def trades(self) -> list[Trade]:
        out, self._trades = self._trades, []
        return out
