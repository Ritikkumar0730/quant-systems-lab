"""Credit risk: structural (Merton), regulatory (Basel IRB / Vasicek) and CDS."""
from .cds import CdsLegs, HazardCurve, bootstrap_hazard, cds_legs, flat_discount
from .irb import (capital_requirement, corporate_correlation, expected_loss, maturity_adjustment,
                  risk_weighted_assets, simulate_default_rates, vasicek_quantile)
from .merton import MertonResult, calibrate_from_equity, merton

__all__ = [
    "CdsLegs", "HazardCurve", "MertonResult", "bootstrap_hazard", "calibrate_from_equity", "capital_requirement",
    "cds_legs", "corporate_correlation", "expected_loss", "flat_discount", "maturity_adjustment", "merton",
    "risk_weighted_assets", "simulate_default_rates", "vasicek_quantile",
]
