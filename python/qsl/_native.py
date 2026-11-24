"""ctypes bindings to the native layers.

``libqsl``    - C11 kernels (normal distribution, BSM, RNG, Monte Carlo, bonds)
``libqslcpp`` - C ABI over the C++20 engines (order book, trees, PDE, Heston)

Both are built by ``make`` into ``build/lib``. Set ``QSL_LIB_DIR`` to load
them from somewhere else.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, Structure, c_double, c_int, c_int32, c_int64, c_size_t, c_uint32, c_uint64, c_void_p
from functools import lru_cache
from pathlib import Path

_EXT = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_REPO_ROOT = Path(__file__).resolve().parents[2]


class NativeLibraryMissing(ImportError):
    """Raised when the shared libraries have not been built."""


def _locate(name: str) -> Path:
    filename = f"lib{name}{_EXT}"
    candidates = []
    if env := os.environ.get("QSL_LIB_DIR"):
        candidates.append(Path(env) / filename)
    candidates.append(_REPO_ROOT / "build" / "lib" / filename)
    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(p.parent) for p in candidates)
    raise NativeLibraryMissing(f"{filename} not found in {searched}; run `make` in {_REPO_ROOT}")


# ------------------------------------------------------------- C structures


class Greeks(Structure):
    _fields_ = [(name, c_double) for name in
                ("price", "delta", "gamma", "vega", "theta", "rho", "dividend_rho", "vanna", "volga")]


class McResult(Structure):
    _fields_ = [("price", c_double), ("std_error", c_double), ("beta", c_double), ("n_samples", c_uint64)]


class Rng(Structure):
    _fields_ = [("s", c_uint64 * 4), ("spare", c_double), ("has_spare", c_int)]


class Date(Structure):
    _fields_ = [("year", c_int32), ("month", c_int32), ("day", c_int32)]


class Bond(Structure):
    _fields_ = [("maturity", Date), ("coupon_rate", c_double), ("face", c_double), ("frequency", c_int),
                ("day_count", c_int), ("end_of_month", c_int)]


class BondMetrics(Structure):
    _fields_ = [("dirty_price", c_double), ("clean_price", c_double), ("accrued", c_double),
                ("macaulay_duration", c_double), ("modified_duration", c_double), ("convexity", c_double),
                ("dv01", c_double), ("cashflows_remaining", c_int32), ("previous_coupon", Date),
                ("next_coupon", Date)]


class CppTrade(Structure):
    _fields_ = [("taker", c_uint64), ("maker", c_uint64), ("price", c_int64), ("qty", c_uint64),
                ("taker_side", c_int32)]


_P_DOUBLE = POINTER(c_double)


def _sig(lib: ctypes.CDLL, name: str, restype, *argtypes) -> None:
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)


@lru_cache(maxsize=None)
def clib() -> ctypes.CDLL:
    """The C kernel library with argument and return types declared."""
    lib = ctypes.CDLL(str(_locate("qsl")))
    d, i = c_double, c_int
    _sig(lib, "qsl_status_str", ctypes.c_char_p, i)
    _sig(lib, "qsl_norm_pdf", d, d)
    _sig(lib, "qsl_norm_cdf", d, d)
    _sig(lib, "qsl_norm_inv", d, d)
    _sig(lib, "qsl_bsm_price", d, i, d, d, d, d, d, d)
    _sig(lib, "qsl_bsm_greeks", i, i, d, d, d, d, d, d, POINTER(Greeks))
    _sig(lib, "qsl_bsm_implied_vol", i, i, d, d, d, d, d, d, _P_DOUBLE)
    _sig(lib, "qsl_bsm_price_batch", None, c_size_t, POINTER(c_int), _P_DOUBLE, _P_DOUBLE, _P_DOUBLE,
         _P_DOUBLE, _P_DOUBLE, _P_DOUBLE, _P_DOUBLE)
    _sig(lib, "qsl_rng_seed", None, POINTER(Rng), c_uint64)
    _sig(lib, "qsl_rng_next", c_uint64, POINTER(Rng))
    _sig(lib, "qsl_rng_uniform", d, POINTER(Rng))
    _sig(lib, "qsl_rng_normal", d, POINTER(Rng))
    _sig(lib, "qsl_mc_european", i, i, d, d, d, d, d, d, c_uint64, ctypes.c_uint, c_uint64, POINTER(McResult))
    _sig(lib, "qsl_mc_asian", i, i, i, d, d, d, d, d, d, c_uint32, c_uint64, ctypes.c_uint, c_uint64,
         POINTER(McResult))
    _sig(lib, "qsl_asian_geometric_price", d, i, d, d, d, d, d, d, c_uint32)
    _sig(lib, "qsl_date_to_serial", c_int64, Date)
    _sig(lib, "qsl_date_from_serial", Date, c_int64)
    _sig(lib, "qsl_year_fraction", d, i, Date, Date)
    _sig(lib, "qsl_year_fraction_icma", d, Date, Date, Date, Date, i)
    _sig(lib, "qsl_bond_price", i, POINTER(Bond), Date, d, i, POINTER(BondMetrics))
    _sig(lib, "qsl_bond_yield", i, POINTER(Bond), Date, d, i, _P_DOUBLE)
    return lib


@lru_cache(maxsize=None)
def cpplib() -> ctypes.CDLL:
    """The C++ engine library with argument and return types declared."""
    lib = ctypes.CDLL(str(_locate("qslcpp")))
    d, i, u64, i64 = c_double, c_int, c_uint64, c_int64
    _sig(lib, "qsl_cpp_binomial", i, i, i, d, d, d, d, d, d, u64, i, _P_DOUBLE)
    _sig(lib, "qsl_cpp_fd", i, i, i, d, d, d, d, d, d, u64, u64, _P_DOUBLE, _P_DOUBLE, _P_DOUBLE)
    _sig(lib, "qsl_cpp_heston", i, i, d, d, d, d, d, d, d, d, d, d, _P_DOUBLE)
    _sig(lib, "qsl_cpp_book_new", c_void_p, i64, i64, c_uint32)
    _sig(lib, "qsl_cpp_book_free", None, c_void_p)
    _sig(lib, "qsl_cpp_book_add", i, c_void_p, u64, i, i64, u64, i, POINTER(u64), POINTER(u64))
    _sig(lib, "qsl_cpp_book_market", i, c_void_p, u64, i, u64, POINTER(u64))
    _sig(lib, "qsl_cpp_book_cancel", i, c_void_p, u64)
    _sig(lib, "qsl_cpp_book_reduce", i, c_void_p, u64, u64)
    _sig(lib, "qsl_cpp_book_replace", i, c_void_p, u64, i64, u64, POINTER(u64), POINTER(u64))
    _sig(lib, "qsl_cpp_book_best", i, c_void_p, i, POINTER(i64))
    _sig(lib, "qsl_cpp_book_volume_at", u64, c_void_p, i, i64)
    _sig(lib, "qsl_cpp_book_order_count", u64, c_void_p)
    _sig(lib, "qsl_cpp_book_depth", c_size_t, c_void_p, i, POINTER(i64), POINTER(u64), POINTER(c_uint32), c_size_t)
    _sig(lib, "qsl_cpp_book_drain_trades", c_size_t, c_void_p, POINTER(CppTrade), c_size_t)
    _sig(lib, "qsl_cpp_book_check", i, c_void_p)
    return lib


def status_message(code: int) -> str:
    return clib().qsl_status_str(code).decode()
