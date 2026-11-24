"""VaR backtesting: Kupiec, Christoffersen and the Basel traffic light.

``exceptions`` is a boolean series: True on days the realised loss exceeded
the VaR forecast. ``p`` is the expected exception rate (1 - confidence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, chi2


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with the convention 0 * log(0) = 0."""
    return 0.0 if x == 0 else x * np.log(y)


def _bernoulli_loglik(failures: int, total: int, p: float) -> float:
    return _xlogy(total - failures, 1 - p) + _xlogy(failures, p)


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    exceptions: int
    observations: int

    def rejects(self, significance: float = 0.05) -> bool:
        return self.p_value < significance


def kupiec_pof(exceptions, p: float = 0.01) -> TestResult:
    """Proportion-of-failures likelihood-ratio test (unconditional coverage),
    LR ~ chi2(1) under H0: exception rate == p."""
    e = np.asarray(exceptions, dtype=bool)
    n, x = e.size, int(e.sum())
    lr = -2.0 * (_bernoulli_loglik(x, n, p) - _bernoulli_loglik(x, n, x / n))
    return TestResult(lr, float(chi2.sf(lr, 1)), x, n)


def christoffersen_independence(exceptions) -> TestResult:
    """Tests whether exceptions cluster (first-order Markov alternative),
    LR ~ chi2(1) under H0: today's exception is independent of yesterday's."""
    e = np.asarray(exceptions, dtype=bool).astype(int)
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    restricted = _bernoulli_loglik(n01 + n11, n00 + n01 + n10 + n11, pi)
    unrestricted = _bernoulli_loglik(n01, n00 + n01, pi01) + _bernoulli_loglik(n11, n10 + n11, pi11)
    lr = max(-2.0 * (restricted - unrestricted), 0.0)
    return TestResult(lr, float(chi2.sf(lr, 1)), int(e.sum()), e.size)


def conditional_coverage(exceptions, p: float = 0.01) -> TestResult:
    """Christoffersen's joint test: LR_cc = LR_pof + LR_ind ~ chi2(2)."""
    pof = kupiec_pof(exceptions, p)
    ind = christoffersen_independence(exceptions)
    lr = pof.statistic + ind.statistic
    return TestResult(lr, float(chi2.sf(lr, 2)), pof.exceptions, pof.observations)


# Basel Committee (1996) plus factors for 250 observations at 99%.
_BASEL_PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


@dataclass(frozen=True)
class TrafficLight:
    zone: str                    # "green", "yellow" or "red"
    cumulative_probability: float  # P(X <= exceptions) under a correct model
    plus_factor: float | None    # add-on to the capital multiplier of 3
    multiplier: float | None


def basel_traffic_light(exceptions: int, observations: int = 250, p: float = 0.01) -> TrafficLight:
    """Zone boundaries from the binomial distribution: green while
    P(X <= x) < 95%, red once it reaches 99.99%. For the regulatory
    250-day / 99% case this gives the familiar 0-4 green, 5-9 yellow,
    10+ red table, and the plus factor is reported."""
    cum = float(binom.cdf(exceptions, observations, p))
    zone = "green" if cum < 0.95 else ("red" if cum >= 0.9999 else "yellow")
    plus = None
    if observations == 250 and p == 0.01:
        plus = 0.0 if zone == "green" else (1.0 if zone == "red" else _BASEL_PLUS[exceptions])
    return TrafficLight(zone, cum, plus, None if plus is None else 3.0 + plus)
