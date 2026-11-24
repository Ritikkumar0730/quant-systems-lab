/*
 * qsl.h - Quant Systems Lab: C11 numerical kernels for pricing and fixed income.
 *
 * Pure C11 + libm, no global state. Every function is thread-safe; the only
 * mutable object is the caller-owned qsl_rng.
 *
 * Conventions
 *   - Rates, yields, dividend yields and volatilities are annualised decimals
 *     (0.05 == 5%). Times are ACT/365-style year fractions.
 *   - q is a continuous dividend yield (or the foreign rate, which turns BSM
 *     into Garman-Kohlhagen for FX options).
 *   - Greeks are per unit move: vega per 1.00 of vol (divide by 100 for "per
 *     vol point"), rho per 1.00 of rate, theta per year (divide by 365 for
 *     "per calendar day").
 */
#ifndef QSL_QSL_H
#define QSL_QSL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QSL_VERSION_MAJOR 1
#define QSL_VERSION_MINOR 0
#define QSL_VERSION_PATCH 0

/* ----------------------------------------------------------------- status */

typedef enum qsl_status {
    QSL_OK = 0,
    QSL_E_INVALID_ARG = 1,   /* NULL pointer, unknown enum value, n == 0        */
    QSL_E_DOMAIN = 2,        /* input outside the model domain (S <= 0, T < 0)  */
    QSL_E_ARBITRAGE = 3,     /* price violates static no-arbitrage bounds       */
    QSL_E_NO_CONVERGENCE = 4 /* iterative solver hit its iteration limit        */
} qsl_status;

const char *qsl_status_str(qsl_status status);

/* ---------------------------------------------------- normal distribution */

double qsl_norm_pdf(double x);
double qsl_norm_cdf(double x);
/* Inverse CDF (probit). -inf/+inf at 0/1, NaN outside [0, 1].
 * Acklam's rational approximation polished by one Halley step (~1e-15 rel). */
double qsl_norm_inv(double p);

/* ------------------------------------------------- Black-Scholes-Merton */

typedef enum qsl_option_type { QSL_PUT = -1, QSL_CALL = 1 } qsl_option_type;

typedef struct qsl_greeks {
    double price;
    double delta;        /* dV/dS                                     */
    double gamma;        /* d2V/dS2                                   */
    double vega;         /* dV/dsigma                                 */
    double theta;        /* dV/dt = -dV/dT, per year                  */
    double rho;          /* dV/dr                                     */
    double dividend_rho; /* dV/dq                                     */
    double vanna;        /* d2V/dS dsigma = d(delta)/d(sigma)         */
    double volga;        /* d2V/dsigma2 (vomma)                       */
} qsl_greeks;

/* Generalised BSM price. Returns NaN on invalid input. sigma*sqrt(t) == 0 is
 * allowed and returns the discounted forward intrinsic value. */
double qsl_bsm_price(qsl_option_type type, double spot, double strike, double t,
                     double r, double q, double sigma);

qsl_status qsl_bsm_greeks(qsl_option_type type, double spot, double strike,
                          double t, double r, double q, double sigma,
                          qsl_greeks *out);

/* Implied volatility: Newton-Raphson on vega, safeguarded by a maintained
 * bisection bracket so it cannot diverge. QSL_E_ARBITRAGE when the price is
 * outside (discounted forward intrinsic, upper bound). */
qsl_status qsl_bsm_implied_vol(qsl_option_type type, double price, double spot,
                               double strike, double t, double r, double q,
                               double *sigma_out);

/* Structure-of-arrays batch pricer; type[i] is +1 (call) or -1 (put).
 * Invalid rows produce NaN. */
void qsl_bsm_price_batch(size_t n, const int *type, const double *spot,
                         const double *strike, const double *t, const double *r,
                         const double *q, const double *sigma, double *out);

/* ------------------------------------------------------------------- RNG */

/* xoshiro256** (Blackman & Vigna) seeded through SplitMix64. */
typedef struct qsl_rng {
    uint64_t s[4];
    double spare;  /* second variate cached by the polar method */
    int has_spare;
} qsl_rng;

void qsl_rng_seed(qsl_rng *rng, uint64_t seed);
uint64_t qsl_rng_next(qsl_rng *rng);
double qsl_rng_uniform(qsl_rng *rng); /* [0, 1) with 53 random bits    */
double qsl_rng_normal(qsl_rng *rng);  /* N(0, 1), Marsaglia polar method */

/* ------------------------------------------------------------ Monte Carlo */

enum {
    QSL_MC_PLAIN = 0u,
    QSL_MC_ANTITHETIC = 1u << 0,     /* pair each draw Z with its mirror -Z      */
    QSL_MC_CONTROL_VARIATE = 1u << 1 /* regression control variate (see below) */
};

typedef enum qsl_average_type {
    QSL_AVG_ARITHMETIC = 0,
    QSL_AVG_GEOMETRIC = 1
} qsl_average_type;

typedef struct qsl_mc_result {
    double price;
    double std_error;   /* standard error of the price estimate             */
    double beta;        /* fitted control-variate coefficient (0 if unused) */
    uint64_t n_samples; /* independent samples (an antithetic pair is one)  */
} qsl_mc_result;

/* European option under GBM, exact terminal sampling. With
 * QSL_MC_CONTROL_VARIATE the control is the discounted terminal spot, whose
 * risk-neutral mean S*exp(-qT) is known exactly. n_paths counts independent
 * Gaussian draws; with QSL_MC_ANTITHETIC each draw prices two mirrored paths. */
qsl_status qsl_mc_european(qsl_option_type type, double spot, double strike,
                           double t, double r, double q, double sigma,
                           uint64_t n_paths, unsigned flags, uint64_t seed,
                           qsl_mc_result *out);

/* Discretely monitored Asian option with n_fixings equally spaced fixings at
 * t_i = i*T/n. For an arithmetic average, QSL_MC_CONTROL_VARIATE uses the
 * geometric-average option (closed form below) as the control: the classic
 * Kemna-Vorst variance-reduction trick. A geometric average with
 * QSL_MC_CONTROL_VARIATE is rejected (the control would be the target). */
qsl_status qsl_mc_asian(qsl_option_type type, qsl_average_type average,
                        double spot, double strike, double t, double r, double q,
                        double sigma, uint32_t n_fixings, uint64_t n_paths,
                        unsigned flags, uint64_t seed, qsl_mc_result *out);

/* Closed-form price of the discretely monitored geometric-average Asian. */
double qsl_asian_geometric_price(qsl_option_type type, double spot, double strike,
                                 double t, double r, double q, double sigma,
                                 uint32_t n_fixings);

/* -------------------------------------------------- dates and day counts */

typedef struct qsl_date {
    int32_t year;
    int32_t month; /* 1..12 */
    int32_t day;   /* 1..31 */
} qsl_date;

int qsl_is_leap_year(int32_t year);
int32_t qsl_days_in_month(int32_t year, int32_t month);
int qsl_date_is_valid(qsl_date d);
/* Days since 1970-01-01 in the proleptic Gregorian calendar. */
int64_t qsl_date_to_serial(qsl_date d);
qsl_date qsl_date_from_serial(int64_t serial);
/* Shift by whole months, clamping the day to the target month. With
 * end_of_month != 0 a month-end input maps to the target month-end. */
qsl_date qsl_date_add_months(qsl_date d, int32_t months, int end_of_month);

typedef enum qsl_day_count {
    QSL_DC_ACT_360 = 0,
    QSL_DC_ACT_365F = 1,
    QSL_DC_30_360_BOND_BASIS = 2, /* ISDA 2006 4.16(f)                         */
    QSL_DC_30E_360 = 3,           /* ISDA 2006 4.16(g), Eurobond basis         */
    QSL_DC_ACT_ACT_ISDA = 4,
    QSL_DC_ACT_ACT_ICMA = 5       /* needs a reference period: see _icma below */
} qsl_day_count;

/* Year fraction from d1 to d2 (negative if d2 < d1). ACT/ACT ICMA returns NaN
 * here because it is only defined against a reference coupon period. */
double qsl_year_fraction(qsl_day_count dc, qsl_date d1, qsl_date d2);
/* ACT/ACT ICMA for d1..d2 inside the regular period [ref_start, ref_end]. */
double qsl_year_fraction_icma(qsl_date d1, qsl_date d2, qsl_date ref_start,
                              qsl_date ref_end, int frequency);

/* ------------------------------------------------------------ fixed income */

typedef enum qsl_compounding {
    QSL_COMPOUNDED = 0, /* periodic at the bond's coupon frequency (street) */
    QSL_CONTINUOUS = 1
} qsl_compounding;

/* Fixed-coupon bullet bond on an unadjusted regular schedule rolled back from
 * maturity. Each coupon pays face * coupon_rate / frequency (market
 * convention for ACT/ACT ICMA government bonds and 30/360 corporates). */
typedef struct qsl_bond {
    qsl_date maturity;
    double coupon_rate;      /* annual, e.g. 0.0725 for a 7.25% coupon        */
    double face;             /* redemption amount, e.g. 100                   */
    int frequency;           /* coupons per year: 1, 2, 4 or 12               */
    qsl_day_count day_count; /* ACT/ACT ICMA, 30/360 bond basis or 30E/360    */
    int end_of_month;        /* keep coupon dates on month ends               */
} qsl_bond;

typedef struct qsl_bond_metrics {
    double dirty_price;       /* full price = clean + accrued                */
    double clean_price;
    double accrued;           /* accrued interest                            */
    double macaulay_duration; /* years                                       */
    double modified_duration; /* -(1/P) dP/dy                                */
    double convexity;         /* (1/P) d2P/dy2                               */
    double dv01;              /* -dP/dy * 1bp, per bond (same units as face) */
    int32_t cashflows_remaining;
    qsl_date previous_coupon;
    qsl_date next_coupon;
} qsl_bond_metrics;

/* Price and risk from a yield, street convention: the first cash flow is
 * discounted over the fraction w of a coupon period still to run. */
qsl_status qsl_bond_price(const qsl_bond *bond, qsl_date settlement, double yield,
                          qsl_compounding compounding, qsl_bond_metrics *out);

/* Yield to maturity from a clean price (safeguarded Newton). */
qsl_status qsl_bond_yield(const qsl_bond *bond, qsl_date settlement,
                          double clean_price, qsl_compounding compounding,
                          double *yield_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QSL_QSL_H */
