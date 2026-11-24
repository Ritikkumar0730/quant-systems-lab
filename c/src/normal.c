/* Standard normal distribution: pdf, cdf and inverse cdf (probit). */
#include "internal.h"

#include <math.h>

const char *qsl_status_str(qsl_status status) {
    switch (status) {
    case QSL_OK: return "ok";
    case QSL_E_INVALID_ARG: return "invalid argument";
    case QSL_E_DOMAIN: return "input outside model domain";
    case QSL_E_ARBITRAGE: return "price violates no-arbitrage bounds";
    case QSL_E_NO_CONVERGENCE: return "solver did not converge";
    }
    return "unknown status";
}

double qsl_norm_pdf(double x) { return QSL_INV_SQRT_2PI * exp(-0.5 * x * x); }

/* erfc keeps full relative accuracy in the lower tail, unlike 1 - erf. */
double qsl_norm_cdf(double x) { return 0.5 * erfc(-x * QSL_SQRT1_2); }

/* Probit for 0 < p <= 0.5 (the lower half, where erfc is accurate). */
static double probit_lower(double p) {
    /* P. J. Acklam's rational approximation, |relative error| < 1.15e-9. */
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                2.445134137142996e+00, 3.754408661907416e+00};
    const double p_low = 0.02425;

    double x;
    if (p < p_low) {
        const double s = sqrt(-2.0 * log(p));
        x = (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) /
            ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1.0);
    } else {
        const double s = p - 0.5;
        const double s2 = s * s;
        x = (((((a[0] * s2 + a[1]) * s2 + a[2]) * s2 + a[3]) * s2 + a[4]) * s2 + a[5]) * s /
            (((((b[0] * s2 + b[1]) * s2 + b[2]) * s2 + b[3]) * s2 + b[4]) * s2 + 1.0);
    }

    /* One Halley step on Phi(x) - p = 0 lifts the result to full double
     * precision. Skipped only where exp(x^2/2) would overflow (p < ~1e-308). */
    if (0.5 * x * x < 700.0) {
        const double e = qsl_norm_cdf(x) - p;
        const double u = e * QSL_SQRT_2PI * exp(0.5 * x * x);
        x = x - u / (1.0 + 0.5 * x * u);
    }
    return x;
}

double qsl_norm_inv(double p) {
    if (isnan(p) || p < 0.0 || p > 1.0) return NAN;
    if (p == 0.0) return -INFINITY;
    if (p == 1.0) return INFINITY;
    /* For p in [0.5, 1], 1 - p is exact (Sterbenz), so the reflection keeps
     * the result exactly antisymmetric and accurate in the upper tail. */
    if (p > 0.5) return -probit_lower(1.0 - p);
    return probit_lower(p);
}
