"""Credit: Merton against Hull's worked example, Basel IRB against the
published risk weight and against Monte Carlo of the one-factor model, CDS
against the credit triangle and exact bootstrap repricing."""
from __future__ import annotations

import numpy as np
import pytest

from qsl.credit import (HazardCurve, bootstrap_hazard, calibrate_from_equity, capital_requirement, cds_legs,
                        corporate_correlation, flat_discount, merton, risk_weighted_assets, simulate_default_rates,
                        vasicek_quantile)


def test_merton_hull_worked_example():
    """Hull: E = 3, sigma_E = 80%, D = 10, r = 5%, T = 1 gives V0 = 12.40,
    sigma_V = 21.23%, default probability N(-d2) = 12.7%, debt worth 9.40
    and an expected loss of about 1.2% of the debt's no-default value."""
    m = calibrate_from_equity(3.0, 0.80, 10.0, 1.0, 0.05)
    assert m.asset_value == pytest.approx(12.40, abs=0.005)
    assert m.asset_vol == pytest.approx(0.2123, abs=5e-5)
    assert m.default_probability == pytest.approx(0.127, abs=5e-4)
    assert m.debt_value == pytest.approx(9.40, abs=0.005)
    assert m.expected_loss_rate == pytest.approx(0.012, abs=5e-4)
    assert m.equity_value == pytest.approx(3.0, rel=1e-10) and m.equity_vol == pytest.approx(0.80, rel=1e-10)


def test_merton_spread_increases_with_leverage():
    spreads = [merton(100, 0.3, d, 5, 0.03).credit_spread for d in (20, 40, 60, 80)]
    assert all(b > a for a, b in zip(spreads, spreads[1:]))


def test_irb_corporate_risk_weight():
    """PD 1%, LGD 45%, M = 2.5: supervisory correlation 19.28% and a risk
    weight of 92.32% (BCBS IRB risk-weight function)."""
    assert corporate_correlation(0.01) == pytest.approx(0.19278, abs=1e-5)
    assert risk_weighted_assets(0.01, 0.45, 1.0) == pytest.approx(0.9232, abs=1e-4)
    assert corporate_correlation(1e-9) == pytest.approx(0.24, abs=1e-6)
    assert corporate_correlation(0.5) == pytest.approx(0.12, abs=1e-6)
    # Capital rises with PD (over the usual range) and with maturity.
    ks = [capital_requirement(pd, 0.45) for pd in (0.0005, 0.002, 0.01, 0.05)]
    assert all(b > a for a, b in zip(ks, ks[1:]))
    assert capital_requirement(0.01, 0.45, 5.0) > capital_requirement(0.01, 0.45, 1.0)


def test_vasicek_quantile_matches_one_factor_monte_carlo():
    pd, rho = 0.02, 0.15
    rates = simulate_default_rates(pd, rho, n_obligors=20_000, n_scenarios=400_000, seed=11)
    assert rates.mean() == pytest.approx(pd, rel=0.01)
    for q in (0.95, 0.99, 0.999):
        assert np.quantile(rates, q) == pytest.approx(vasicek_quantile(pd, rho, q), rel=0.03)


def test_cds_credit_triangle_and_bootstrap():
    flat = HazardCurve((5.0,), (0.02,))
    # With zero rates, premium + accrual-on-default integrate the survival
    # curve exactly, so the triangle s = lambda (1 - R) holds to quadrature error.
    assert cds_legs(flat, 5.0, 0.4, flat_discount(0.0)).par_spread == pytest.approx(0.012, rel=1e-5)
    # With r > 0, paying premium at period end costs ~ r * tau / 2 extra.
    df = flat_discount(0.04)
    spread = cds_legs(flat, 5.0, 0.4, df).par_spread
    assert spread / 0.012 - 1 == pytest.approx(0.04 * 0.25 / 2, abs=1e-3)

    maturities = [1, 3, 5, 7, 10]
    quotes = [0.0060, 0.0085, 0.0110, 0.0125, 0.0140]
    curve = bootstrap_hazard(maturities, quotes, 0.4, df)
    for t, s in zip(maturities, quotes):
        assert cds_legs(curve, t, 0.4, df).par_spread == pytest.approx(s, abs=1e-12)
    assert all(h > 0 for h in curve.hazards)
    surv = [curve.survival(t) for t in np.linspace(0, 12, 50)]
    assert surv[0] == 1.0 and all(b < a for a, b in zip(surv, surv[1:]))
