"""Fixed-coupon bond analytics backed by the C kernel.

Clean/dirty price, accrued interest, yield to maturity, Macaulay and
modified duration, convexity and DV01 on real calendar dates with
ACT/ACT ICMA, 30/360 bond basis or 30E/360 accrual.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from typing import Literal

from .._native import Bond as _Bond
from .._native import BondMetrics as _BondMetrics
from .._native import Date as _Date
from .._native import clib, status_message

Compounding = Literal["periodic", "continuous"]


class DayCount(IntEnum):
    ACT_360 = 0
    ACT_365F = 1
    THIRTY_360_BOND_BASIS = 2
    THIRTY_E_360 = 3
    ACT_ACT_ISDA = 4
    ACT_ACT_ICMA = 5


def _to_c(d: date) -> _Date:
    return _Date(d.year, d.month, d.day)


def _from_c(d: _Date) -> date:
    return date(d.year, d.month, d.day)


def year_fraction(day_count: DayCount, start: date, end: date) -> float:
    """Accrual year fraction (ACT/ACT ICMA needs a coupon period: use a Bond)."""
    return clib().qsl_year_fraction(int(day_count), _to_c(start), _to_c(end))


@dataclass(frozen=True)
class BondAnalytics:
    dirty_price: float
    clean_price: float
    accrued: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    cashflows_remaining: int
    previous_coupon: date
    next_coupon: date


@dataclass(frozen=True)
class FixedRateBond:
    """Bullet bond paying face * coupon_rate / frequency on a regular schedule
    rolled back from maturity (e.g. a US Treasury: frequency 2, ACT/ACT ICMA;
    an Indian G-Sec: frequency 2, 30/360)."""

    maturity: date
    coupon_rate: float
    frequency: int = 2
    face: float = 100.0
    day_count: DayCount = DayCount.ACT_ACT_ICMA
    end_of_month: bool = False

    def _c(self) -> _Bond:
        return _Bond(_to_c(self.maturity), self.coupon_rate, self.face, self.frequency, int(self.day_count),
                     int(self.end_of_month))

    def analytics(self, settlement: date, ytm: float, compounding: Compounding = "periodic") -> BondAnalytics:
        """Price and risk at a yield (street convention for the first period)."""
        out = _BondMetrics()
        code = clib().qsl_bond_price(ctypes.byref(self._c()), _to_c(settlement), ytm,
                                     0 if compounding == "periodic" else 1, ctypes.byref(out))
        if code:
            raise ValueError(f"bond analytics: {status_message(code)}")
        return BondAnalytics(out.dirty_price, out.clean_price, out.accrued, out.macaulay_duration,
                             out.modified_duration, out.convexity, out.dv01, out.cashflows_remaining,
                             _from_c(out.previous_coupon), _from_c(out.next_coupon))

    def yield_to_maturity(self, settlement: date, clean_price: float, compounding: Compounding = "periodic") -> float:
        out = ctypes.c_double()
        code = clib().qsl_bond_yield(ctypes.byref(self._c()), _to_c(settlement), clean_price,
                                     0 if compounding == "periodic" else 1, ctypes.byref(out))
        if code:
            raise ValueError(f"bond yield: {status_message(code)}")
        return out.value
