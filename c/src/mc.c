/* Monte Carlo pricing under GBM with antithetic and control variates. */
#include "internal.h"

#include <math.h>

/* Welford-style online moments for a (target Y, control X) pair stream.
 * Numerically stable, single pass, no storage of samples. */
typedef struct moments {
    double n;
    double mean_y;
    double mean_x;
    double c_yy; /* sum of squared deviations of Y        */
    double c_xx; /* sum of squared deviations of X        */
    double c_xy; /* sum of cross deviations of X and Y    */
} moments;

static void moments_push(moments *m, double y, double x) {
    m->n += 1.0;
    const double dy = y - m->mean_y;
    const double dx = x - m->mean_x;
    m->mean_y += dy / m->n;
    m->mean_x += dx / m->n;
    m->c_yy += dy * (y - m->mean_y);
    m->c_xx += dx * (x - m->mean_x);
    m->c_xy += dx * (y - m->mean_y);
}

/* Plain estimator, or regression control variate Y - beta (X - E[X]) with the
 * variance-minimising beta = Cov(X, Y) / Var(X). */
static void moments_finish(const moments *m, int use_cv, double control_mean,
                           qsl_mc_result *out) {
    out->n_samples = (uint64_t)m->n;
    if (!use_cv || m->c_xx <= 0.0) {
        out->price = m->mean_y;
        out->std_error = m->n > 1.0 ? sqrt(m->c_yy / (m->n - 1.0) / m->n) : NAN;
        out->beta = 0.0;
        return;
    }
    const double beta = m->c_xy / m->c_xx;
    const double resid = fmax(m->c_yy - beta * m->c_xy, 0.0); /* = c_yy (1 - rho^2) */
    out->price = m->mean_y - beta * (m->mean_x - control_mean);
    out->std_error = m->n > 2.0 ? sqrt(resid / (m->n - 2.0) / m->n) : NAN;
    out->beta = beta;
}

static int valid_gbm(double spot, double strike, double t, double r, double q, double sigma) {
    return isfinite(spot) && isfinite(strike) && isfinite(t) && isfinite(r) &&
           isfinite(q) && isfinite(sigma) && spot > 0.0 && strike > 0.0 && t > 0.0 &&
           sigma >= 0.0;
}

qsl_status qsl_mc_european(qsl_option_type type, double spot, double strike,
                           double t, double r, double q, double sigma,
                           uint64_t n_paths, unsigned flags, uint64_t seed,
                           qsl_mc_result *out) {
    if (out == NULL || !qsl_valid_type(type) || n_paths < 3 ||
        (flags & ~(unsigned)(QSL_MC_ANTITHETIC | QSL_MC_CONTROL_VARIATE)) != 0u)
        return QSL_E_INVALID_ARG;
    if (!valid_gbm(spot, strike, t, r, q, sigma)) return QSL_E_DOMAIN;

    const double phi = (double)type;
    const double df = exp(-r * t);
    const double drift = (r - q - 0.5 * sigma * sigma) * t;
    const double vol = sigma * sqrt(t);
    const int antithetic = (flags & QSL_MC_ANTITHETIC) != 0u;
    const int use_cv = (flags & QSL_MC_CONTROL_VARIATE) != 0u;

    qsl_rng rng;
    qsl_rng_seed(&rng, seed);
    moments m = {0};
    for (uint64_t i = 0; i < n_paths; ++i) {
        const double z = qsl_rng_normal(&rng);
        const double s_up = spot * exp(drift + vol * z);
        double y = df * fmax(phi * (s_up - strike), 0.0);
        double x = df * s_up;
        if (antithetic) {
            const double s_dn = spot * exp(drift - vol * z);
            y = 0.5 * (y + df * fmax(phi * (s_dn - strike), 0.0));
            x = 0.5 * (x + df * s_dn);
        }
        moments_push(&m, y, x);
    }
    /* E[e^{-rT} S_T] = S e^{-qT} under the risk-neutral measure. */
    moments_finish(&m, use_cv, spot * exp(-q * t), out);
    return QSL_OK;
}

double qsl_asian_geometric_price(qsl_option_type type, double spot, double strike,
                                 double t, double r, double q, double sigma,
                                 uint32_t n_fixings) {
    if (!qsl_valid_type(type) || n_fixings == 0u || !valid_gbm(spot, strike, t, r, q, sigma))
        return NAN;
    /* log G is Gaussian: mean and variance of the average of log S(t_i),
     * t_i = i T / n, using sum_{i,j} min(i, j) = n (n + 1)(2n + 1) / 6. */
    const double n = (double)n_fixings;
    const double phi = (double)type;
    const double mean = log(spot) + (r - q - 0.5 * sigma * sigma) * t * (n + 1.0) / (2.0 * n);
    const double var = sigma * sigma * t * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
    const double df = exp(-r * t);
    if (var == 0.0) return df * fmax(phi * (exp(mean) - strike), 0.0);
    const double sd = sqrt(var);
    const double d1 = (mean - log(strike) + var) / sd;
    const double d2 = d1 - sd;
    return df * phi *
           (exp(mean + 0.5 * var) * qsl_norm_cdf(phi * d1) - strike * qsl_norm_cdf(phi * d2));
}

/* Arithmetic and geometric means of one path's fixings. */
typedef struct path_avgs {
    double arithmetic;
    double geometric;
} path_avgs;

qsl_status qsl_mc_asian(qsl_option_type type, qsl_average_type average,
                        double spot, double strike, double t, double r, double q,
                        double sigma, uint32_t n_fixings, uint64_t n_paths,
                        unsigned flags, uint64_t seed, qsl_mc_result *out) {
    if (out == NULL || !qsl_valid_type(type) || n_fixings == 0u || n_paths < 3 ||
        (average != QSL_AVG_ARITHMETIC && average != QSL_AVG_GEOMETRIC) ||
        (flags & ~(unsigned)(QSL_MC_ANTITHETIC | QSL_MC_CONTROL_VARIATE)) != 0u)
        return QSL_E_INVALID_ARG;
    const int use_cv = (flags & QSL_MC_CONTROL_VARIATE) != 0u;
    if (use_cv && average == QSL_AVG_GEOMETRIC) return QSL_E_INVALID_ARG;
    if (!valid_gbm(spot, strike, t, r, q, sigma)) return QSL_E_DOMAIN;

    const double phi = (double)type;
    const double n = (double)n_fixings;
    const double dt = t / n;
    const double step_drift = (r - q - 0.5 * sigma * sigma) * dt;
    const double step_vol = sigma * sqrt(dt);
    const double df = exp(-r * t);
    const double log_spot = log(spot);
    const int antithetic = (flags & QSL_MC_ANTITHETIC) != 0u;

    qsl_rng rng;
    qsl_rng_seed(&rng, seed);
    moments m = {0};
    for (uint64_t p = 0; p < n_paths; ++p) {
        double log_up = log_spot, log_dn = log_spot;
        double sum_up = 0.0, sum_dn = 0.0;         /* sums of fixings     */
        double sum_log_up = 0.0, sum_log_dn = 0.0; /* sums of log fixings */
        for (uint32_t i = 0; i < n_fixings; ++i) {
            const double z = qsl_rng_normal(&rng);
            log_up += step_drift + step_vol * z;
            sum_up += exp(log_up);
            sum_log_up += log_up;
            if (antithetic) {
                log_dn += step_drift - step_vol * z;
                sum_dn += exp(log_dn);
                sum_log_dn += log_dn;
            }
        }
        const path_avgs up = {sum_up / n, exp(sum_log_up / n)};
        const double target_avg_up = average == QSL_AVG_ARITHMETIC ? up.arithmetic : up.geometric;
        double y = df * fmax(phi * (target_avg_up - strike), 0.0);
        double x = df * fmax(phi * (up.geometric - strike), 0.0);
        if (antithetic) {
            const path_avgs dn = {sum_dn / n, exp(sum_log_dn / n)};
            const double target_avg_dn =
                average == QSL_AVG_ARITHMETIC ? dn.arithmetic : dn.geometric;
            y = 0.5 * (y + df * fmax(phi * (target_avg_dn - strike), 0.0));
            x = 0.5 * (x + df * fmax(phi * (dn.geometric - strike), 0.0));
        }
        moments_push(&m, y, x);
    }
    const double control_mean =
        use_cv ? qsl_asian_geometric_price(type, spot, strike, t, r, q, sigma, n_fixings) : 0.0;
    moments_finish(&m, use_cv, control_mean, out);
    return QSL_OK;
}
