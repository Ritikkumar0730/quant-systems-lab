"""Option pricing: Python front end to the C kernels and C++ engines.

Closed form, Greeks and implied vol come from the C library; trees, finite
differences and Heston from the C++ library. ``black_scholes_numpy`` is an
independent pure-NumPy implementation kept for cross-checking and for
benchmarking the C batch kernel against vectorised NumPy.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr

from ._native import Greeks as _Greeks
from ._native import McResult as _McResult
from ._native import clib, cpplib, status_message

CALL = 1
PUT = -1
OptionKind = Literal["call", "put"]

_MC_ANTITHETIC = 1
_MC_CONTROL_VARIATE = 2


def _sign(kind: OptionKind | int) -> int:
    if kind in ("call", CALL):
        return CALL
    if kind in ("put", PUT):
        return PUT
    raise ValueError(f"option kind must be 'call' or 'put', got {kind!r}")


def _check(code: int, what: str) -> None:
    if code != 0:
        raise ValueError(f"{what}: {status_message(code)}")


# ---------------------------------------------------------------- closed form


def black_scholes(kind: OptionKind, spot, strike, t, r, q, sigma):
    """Generalised Black-Scholes-Merton price (C batch kernel, broadcasting).

    Returns a float for scalar inputs, otherwise an ndarray. Invalid rows are NaN.
    """
    args = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (spot, strike, t, r, q, sigma)))
    shape = args[0].shape
    flat = [np.ascontiguousarray(a.ravel()) for a in args]
    types = np.full(flat[0].size, _sign(kind), dtype=np.intc)
    out = np.empty(flat[0].size)
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # noqa: E731
    clib().qsl_bsm_price_batch(out.size, types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                               *(ptr(a) for a in flat), ptr(out))
    return float(out[0]) if shape == () else out.reshape(shape)


def black_scholes_numpy(kind: OptionKind, spot, strike, t, r, q, sigma):
    """Independent NumPy/SciPy implementation (reference and benchmark baseline)."""
    phi = float(_sign(kind))
    spot, strike, t, r, q, sigma = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (spot, strike, t, r, q, sigma)))
    sd = sigma * np.sqrt(t)
    fwd_spot = spot * np.exp(-q * t)
    pv_strike = strike * np.exp(-r * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(fwd_spot / pv_strike) / sd + 0.5 * sd
    price = phi * (fwd_spot * ndtr(phi * d1) - pv_strike * ndtr(phi * (d1 - sd)))
    intrinsic = np.maximum(phi * (fwd_spot - pv_strike), 0.0)
    out = np.where(sd > 0, price, intrinsic)
    return float(out) if out.shape == () else out


@dataclass(frozen=True)
class OptionGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    dividend_rho: float
    vanna: float
    volga: float


def greeks(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, sigma: float) -> OptionGreeks:
    """Price and first/second-order Greeks (per unit moves; theta per year)."""
    g = _Greeks()
    _check(clib().qsl_bsm_greeks(_sign(kind), spot, strike, t, r, q, sigma, ctypes.byref(g)), "greeks")
    return OptionGreeks(**{name: getattr(g, name) for name, _ in _Greeks._fields_})


def implied_vol(kind: OptionKind, price: float, spot: float, strike: float, t: float, r: float, q: float = 0.0) -> float:
    """Black-Scholes implied volatility; raises ValueError outside no-arbitrage bounds."""
    out = ctypes.c_double()
    _check(clib().qsl_bsm_implied_vol(_sign(kind), price, spot, strike, t, r, q, ctypes.byref(out)), "implied vol")
    return out.value


# ---------------------------------------------------------------- Monte Carlo


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    std_error: float
    beta: float
    n_samples: int

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% normal-approximation interval."""
        return self.price - 1.959963984540054 * self.std_error, self.price + 1.959963984540054 * self.std_error


def _flags(antithetic: bool, control_variate: bool) -> int:
    return (_MC_ANTITHETIC if antithetic else 0) | (_MC_CONTROL_VARIATE if control_variate else 0)


def monte_carlo_european(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, sigma: float,
                         n_paths: int = 100_000, *, antithetic: bool = False, control_variate: bool = False,
                         seed: int = 42) -> MonteCarloResult:
    res = _McResult()
    _check(clib().qsl_mc_european(_sign(kind), spot, strike, t, r, q, sigma, n_paths,
                                  _flags(antithetic, control_variate), seed, ctypes.byref(res)), "monte carlo")
    return MonteCarloResult(res.price, res.std_error, res.beta, res.n_samples)


def monte_carlo_asian(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, sigma: float,
                      n_fixings: int, n_paths: int = 100_000, *, average: Literal["arithmetic", "geometric"] = "arithmetic",
                      antithetic: bool = False, control_variate: bool = False, seed: int = 42) -> MonteCarloResult:
    avg = {"arithmetic": 0, "geometric": 1}[average]
    res = _McResult()
    _check(clib().qsl_mc_asian(_sign(kind), avg, spot, strike, t, r, q, sigma, n_fixings, n_paths,
                               _flags(antithetic, control_variate), seed, ctypes.byref(res)), "asian monte carlo")
    return MonteCarloResult(res.price, res.std_error, res.beta, res.n_samples)


def asian_geometric(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, sigma: float,
                    n_fixings: int) -> float:
    """Closed-form discretely monitored geometric-average Asian option."""
    value = clib().qsl_asian_geometric_price(_sign(kind), spot, strike, t, r, q, sigma, n_fixings)
    if np.isnan(value):
        raise ValueError("asian_geometric: invalid input")
    return value


# ------------------------------------------------------- C++ numerical engines


def binomial(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, sigma: float,
             steps: int = 501, *, american: bool = False, tree: Literal["leisen-reimer", "crr"] = "leisen-reimer") -> float:
    """Binomial tree price (Leisen-Reimer by default; CRR for the textbook tree)."""
    out = ctypes.c_double()
    code = cpplib().qsl_cpp_binomial(int(_sign(kind) == CALL), int(american), spot, strike, t, r, q, sigma, steps,
                                     int(tree == "leisen-reimer"), ctypes.byref(out))
    if code:
        raise ValueError("binomial: invalid input")
    return out.value


@dataclass(frozen=True)
class FdResult:
    price: float
    delta: float
    gamma: float


def finite_difference(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, sigma: float, *,
                      american: bool = False, space_steps: int = 400, time_steps: int = 400) -> FdResult:
    """Crank-Nicolson PDE price with grid delta and gamma."""
    p, d, g = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    code = cpplib().qsl_cpp_fd(int(_sign(kind) == CALL), int(american), spot, strike, t, r, q, sigma, space_steps,
                               time_steps, ctypes.byref(p), ctypes.byref(d), ctypes.byref(g))
    if code:
        raise ValueError("finite_difference: invalid input")
    return FdResult(p.value, d.value, g.value)


@dataclass(frozen=True)
class HestonParams:
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # spot-variance correlation


def heston(kind: OptionKind, spot: float, strike: float, t: float, r: float, q: float, params: HestonParams) -> float:
    """Semi-analytic Heston price (characteristic-function inversion)."""
    out = ctypes.c_double()
    code = cpplib().qsl_cpp_heston(int(_sign(kind) == CALL), spot, strike, t, r, q, params.v0, params.kappa,
                                   params.theta, params.xi, params.rho, ctypes.byref(out))
    if code:
        raise ValueError("heston: invalid parameters")
    return out.value
