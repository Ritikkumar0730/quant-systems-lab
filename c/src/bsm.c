/* Generalised Black-Scholes-Merton: price, Greeks and implied volatility. */
#include "internal.h"

#include <float.h>
#include <math.h>

static int finite_inputs(double spot, double strike, double t, double r, double q,
                         double sigma) {
    return isfinite(spot) && isfinite(strike) && isfinite(t) && isfinite(r) &&
           isfinite(q) && isfinite(sigma);
}

static int in_domain(double spot, double strike, double t, double r, double q,
                     double sigma) {
    return finite_inputs(spot, strike, t, r, q, sigma) && spot > 0.0 && strike > 0.0 &&
           t >= 0.0 && sigma >= 0.0;
}

double qsl_bsm_price(qsl_option_type type, double spot, double strike, double t,
                     double r, double q, double sigma) {
    if (!qsl_valid_type(type) || !in_domain(spot, strike, t, r, q, sigma)) return NAN;

    const double phi = (double)type;
    const double fwd_spot = spot * exp(-q * t);   /* S e^{-qT} */
    const double pv_strike = strike * exp(-r * t); /* K e^{-rT} */
    const double sd = sigma * sqrt(t);
    if (sd == 0.0) return fmax(phi * (fwd_spot - pv_strike), 0.0);

    const double d1 = log(fwd_spot / pv_strike) / sd + 0.5 * sd;
    const double d2 = d1 - sd;
    return phi * (fwd_spot * qsl_norm_cdf(phi * d1) - pv_strike * qsl_norm_cdf(phi * d2));
}

qsl_status qsl_bsm_greeks(qsl_option_type type, double spot, double strike,
                          double t, double r, double q, double sigma,
                          qsl_greeks *out) {
    if (out == NULL || !qsl_valid_type(type)) return QSL_E_INVALID_ARG;
    if (!in_domain(spot, strike, t, r, q, sigma)) return QSL_E_DOMAIN;

    const double phi = (double)type;
    const double df_q = exp(-q * t);
    const double df_r = exp(-r * t);
    const double sqrt_t = sqrt(t);
    const double sd = sigma * sqrt_t;
    *out = (qsl_greeks){0};

    if (sd == 0.0) {
        /* Deterministic forward: the option is a forward contract or nothing.
         * Second-order Greeks are zero away from the kink (undefined on it). */
        const double forward_value = phi * (spot * df_q - strike * df_r);
        if (forward_value > 0.0) {
            out->price = forward_value;
            out->delta = phi * df_q;
            out->theta = phi * (q * spot * df_q - r * strike * df_r);
            out->rho = phi * strike * t * df_r;
            out->dividend_rho = -phi * spot * t * df_q;
        }
        return QSL_OK;
    }

    const double d1 = log(spot * df_q / (strike * df_r)) / sd + 0.5 * sd;
    const double d2 = d1 - sd;
    const double n_d1 = qsl_norm_pdf(d1);
    const double cdf_d1 = qsl_norm_cdf(phi * d1);
    const double cdf_d2 = qsl_norm_cdf(phi * d2);

    out->price = phi * (spot * df_q * cdf_d1 - strike * df_r * cdf_d2);
    out->delta = phi * df_q * cdf_d1;
    out->gamma = df_q * n_d1 / (spot * sd);
    out->vega = spot * df_q * n_d1 * sqrt_t;
    out->theta = -spot * df_q * n_d1 * sigma / (2.0 * sqrt_t) -
                 phi * r * strike * df_r * cdf_d2 + phi * q * spot * df_q * cdf_d1;
    out->rho = phi * strike * t * df_r * cdf_d2;
    out->dividend_rho = -phi * spot * t * df_q * cdf_d1;
    out->vanna = -df_q * n_d1 * d2 / sigma;
    out->volga = out->vega * d1 * d2 / sigma;
    return QSL_OK;
}

/* Price and vega in one pass for the implied-vol iteration. */
static void price_and_vega(double phi, double fwd_spot, double pv_strike, double sqrt_t,
                           double sigma, double *price, double *vega) {
    const double sd = sigma * sqrt_t;
    if (sd <= 0.0) {
        *price = fmax(phi * (fwd_spot - pv_strike), 0.0);
        *vega = 0.0;
        return;
    }
    const double d1 = log(fwd_spot / pv_strike) / sd + 0.5 * sd;
    const double d2 = d1 - sd;
    *price = phi * (fwd_spot * qsl_norm_cdf(phi * d1) - pv_strike * qsl_norm_cdf(phi * d2));
    *vega = fwd_spot * qsl_norm_pdf(d1) * sqrt_t;
}

qsl_status qsl_bsm_implied_vol(qsl_option_type type, double price, double spot,
                               double strike, double t, double r, double q,
                               double *sigma_out) {
    if (sigma_out == NULL || !qsl_valid_type(type)) return QSL_E_INVALID_ARG;
    *sigma_out = NAN;
    if (!finite_inputs(spot, strike, t, r, q, 0.0) || !isfinite(price) || spot <= 0.0 ||
        strike <= 0.0 || t <= 0.0)
        return QSL_E_DOMAIN;

    const double phi = (double)type;
    const double fwd_spot = spot * exp(-q * t);
    const double pv_strike = strike * exp(-r * t);
    const double sqrt_t = sqrt(t);

    /* Static no-arbitrage bounds: intrinsic value on the forward below, the
     * discounted underlying (call) or discounted strike (put) above. */
    const double lower = fmax(phi * (fwd_spot - pv_strike), 0.0);
    const double upper = type == QSL_CALL ? fwd_spot : pv_strike;
    const double eps = 8.0 * DBL_EPSILON * upper;
    if (price < lower - eps || price >= upper) return QSL_E_ARBITRAGE;
    if (price <= lower + eps) {
        *sigma_out = 0.0;
        return QSL_OK;
    }

    /* BSM price is strictly increasing in sigma: bracket the root first. */
    double lo = 0.0;
    double hi = 1.0;
    double p_hi;
    double v_hi;
    price_and_vega(phi, fwd_spot, pv_strike, sqrt_t, hi, &p_hi, &v_hi);
    while (p_hi < price) {
        lo = hi;
        hi *= 2.0;
        if (hi > 1.0e4) return QSL_E_NO_CONVERGENCE;
        price_and_vega(phi, fwd_spot, pv_strike, sqrt_t, hi, &p_hi, &v_hi);
    }

    /* Manaster-Koehler start (monotone Newton convergence for BSM); near the
     * money it degenerates to 0, so fall back to Brenner-Subrahmanyam. */
    double sigma = sqrt(2.0 * fabs(log(fwd_spot / pv_strike)) / t);
    if (sigma < 1e-3) sigma = QSL_SQRT_2PI * price / (fwd_spot * sqrt_t);
    if (!(sigma > lo && sigma < hi)) sigma = 0.5 * (lo + hi);

    for (int iter = 0; iter < 100; ++iter) {
        double model;
        double vega;
        price_and_vega(phi, fwd_spot, pv_strike, sqrt_t, sigma, &model, &vega);
        const double diff = model - price;
        if (fabs(diff) <= 4.0 * DBL_EPSILON * upper) {
            *sigma_out = sigma;
            return QSL_OK;
        }
        if (diff > 0.0) hi = sigma;
        else lo = sigma;

        double next = vega > 0.0 ? sigma - diff / vega : lo - 1.0;
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi); /* bisection fallback */
        if (fabs(next - sigma) <= 2.0 * DBL_EPSILON * sigma || hi - lo <= 2.0 * DBL_EPSILON * hi) {
            *sigma_out = next;
            return QSL_OK;
        }
        sigma = next;
    }
    return QSL_E_NO_CONVERGENCE;
}

void qsl_bsm_price_batch(size_t n, const int *type, const double *spot,
                         const double *strike, const double *t, const double *r,
                         const double *q, const double *sigma, double *out) {
    if (out == NULL || type == NULL || spot == NULL || strike == NULL || t == NULL ||
        r == NULL || q == NULL || sigma == NULL)
        return;
    for (size_t i = 0; i < n; ++i) {
        out[i] = qsl_bsm_price((qsl_option_type)type[i], spot[i], strike[i], t[i], r[i],
                               q[i], sigma[i]);
    }
}
