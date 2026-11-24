/* Fixed-coupon bond analytics: price, accrued, yield, duration, convexity, DV01. */
#include "internal.h"

#include <float.h>
#include <math.h>

#define QSL_MAX_PERIODS 12000 /* 1000 years of monthly coupons: corrupt-input guard */

static int valid_bond(const qsl_bond *b) {
    return b != NULL && qsl_date_is_valid(b->maturity) &&
           (b->frequency == 1 || b->frequency == 2 || b->frequency == 4 || b->frequency == 12) &&
           isfinite(b->coupon_rate) && b->coupon_rate >= 0.0 && isfinite(b->face) &&
           b->face > 0.0 &&
           (b->day_count == QSL_DC_ACT_ACT_ICMA || b->day_count == QSL_DC_30_360_BOND_BASIS ||
            b->day_count == QSL_DC_30E_360);
}

/* Where the settlement date sits inside the coupon schedule. */
typedef struct bond_position {
    qsl_date previous;       /* last coupon date on or before settlement       */
    qsl_date next;           /* first coupon date after settlement             */
    int32_t n_cashflows;     /* coupons still to be paid                       */
    double accrued_fraction; /* share of the current period already accrued    */
    double w;                /* coupon periods from settlement to `next`       */
} bond_position;

static qsl_status locate(const qsl_bond *b, qsl_date settlement, bond_position *pos) {
    const int64_t s = qsl_date_to_serial(settlement);
    if (s >= qsl_date_to_serial(b->maturity)) return QSL_E_DOMAIN;

    /* Coupon date k is maturity - k * (12 / f) months, always rolled from
     * maturity (not iteratively) so month-end clamping cannot drift. */
    const int32_t step = 12 / b->frequency;
    int32_t k = 1;
    qsl_date next = b->maturity;
    qsl_date prev = qsl_date_add_months(b->maturity, -step, b->end_of_month);
    while (qsl_date_to_serial(prev) > s) {
        if (++k > QSL_MAX_PERIODS) return QSL_E_DOMAIN;
        next = prev;
        prev = qsl_date_add_months(b->maturity, -step * k, b->end_of_month);
    }
    pos->previous = prev;
    pos->next = next;
    pos->n_cashflows = k;

    if (b->day_count == QSL_DC_ACT_ACT_ICMA) {
        const double period = (double)(qsl_date_to_serial(next) - qsl_date_to_serial(prev));
        pos->accrued_fraction = (double)(s - qsl_date_to_serial(prev)) / period;
        pos->w = (double)(qsl_date_to_serial(next) - s) / period;
    } else {
        const double f = (double)b->frequency;
        pos->accrued_fraction = f * qsl_year_fraction(b->day_count, prev, settlement);
        pos->w = f * qsl_year_fraction(b->day_count, settlement, next);
    }
    return QSL_OK;
}

/* Sums over remaining cash flows CF_k paid e_k = k + w periods from now
 * (t_k = e_k / f years): PV, sum CF t DF, and the second-moment sum used by
 * convexity (t (t + 1/f) for periodic compounding, t^2 for continuous). */
typedef struct pv_sums {
    double pv;
    double pv_t;
    double pv_tt;
} pv_sums;

static pv_sums discount(const qsl_bond *b, const bond_position *pos, double y,
                        qsl_compounding comp) {
    const double f = (double)b->frequency;
    const double coupon = b->face * b->coupon_rate / f;
    double df;
    double step_df;
    if (comp == QSL_COMPOUNDED) {
        df = pow(1.0 + y / f, -pos->w);
        step_df = 1.0 / (1.0 + y / f);
    } else {
        df = exp(-y * pos->w / f);
        step_df = exp(-y / f);
    }
    pv_sums sums = {0.0, 0.0, 0.0};
    for (int32_t k = 0; k < pos->n_cashflows; ++k) {
        const double t = ((double)k + pos->w) / f;
        const double cf = coupon + (k == pos->n_cashflows - 1 ? b->face : 0.0);
        const double pv = cf * df;
        sums.pv += pv;
        sums.pv_t += pv * t;
        sums.pv_tt += comp == QSL_COMPOUNDED ? pv * t * (t + 1.0 / f) : pv * t * t;
        df *= step_df;
    }
    return sums;
}

static int valid_yield(double y, int frequency, qsl_compounding comp) {
    return isfinite(y) && (comp == QSL_CONTINUOUS || 1.0 + y / (double)frequency > 0.0);
}

qsl_status qsl_bond_price(const qsl_bond *bond, qsl_date settlement, double yield,
                          qsl_compounding compounding, qsl_bond_metrics *out) {
    if (out == NULL || !valid_bond(bond) || !qsl_date_is_valid(settlement) ||
        (compounding != QSL_COMPOUNDED && compounding != QSL_CONTINUOUS))
        return QSL_E_INVALID_ARG;
    if (!valid_yield(yield, bond->frequency, compounding)) return QSL_E_DOMAIN;

    bond_position pos;
    const qsl_status st = locate(bond, settlement, &pos);
    if (st != QSL_OK) return st;

    const double f = (double)bond->frequency;
    const pv_sums s = discount(bond, &pos, yield, compounding);
    const double growth = compounding == QSL_COMPOUNDED ? 1.0 + yield / f : 1.0;

    out->dirty_price = s.pv;
    out->accrued = bond->face * bond->coupon_rate / f * pos.accrued_fraction;
    out->clean_price = s.pv - out->accrued;
    out->macaulay_duration = s.pv_t / s.pv;
    out->modified_duration = out->macaulay_duration / growth;
    out->convexity = s.pv_tt / (s.pv * growth * growth);
    out->dv01 = out->modified_duration * s.pv * 1e-4;
    out->cashflows_remaining = pos.n_cashflows;
    out->previous_coupon = pos.previous;
    out->next_coupon = pos.next;
    return QSL_OK;
}

qsl_status qsl_bond_yield(const qsl_bond *bond, qsl_date settlement,
                          double clean_price, qsl_compounding compounding,
                          double *yield_out) {
    if (yield_out == NULL || !valid_bond(bond) || !qsl_date_is_valid(settlement) ||
        (compounding != QSL_COMPOUNDED && compounding != QSL_CONTINUOUS))
        return QSL_E_INVALID_ARG;
    *yield_out = NAN;
    if (!isfinite(clean_price) || clean_price <= 0.0) return QSL_E_DOMAIN;

    bond_position pos;
    const qsl_status st = locate(bond, settlement, &pos);
    if (st != QSL_OK) return st;

    const double f = (double)bond->frequency;
    const double target = clean_price + bond->face * bond->coupon_rate / f * pos.accrued_fraction;

    /* Dirty price is strictly decreasing in the yield: bracket, then run
     * Newton with dP/dy = -(PV_t / growth), falling back to bisection. */
    const double floor_y = compounding == QSL_COMPOUNDED ? -f : -INFINITY;
    double lo = -0.5;
    for (int i = 0; discount(bond, &pos, lo, compounding).pv < target; ++i) {
        if (i == 200) return QSL_E_NO_CONVERGENCE;
        lo = compounding == QSL_COMPOUNDED ? 0.5 * (lo + floor_y) : lo - 1.0;
    }
    double hi = 1.0;
    for (int i = 0; discount(bond, &pos, hi, compounding).pv > target; ++i) {
        if (i == 60) return QSL_E_NO_CONVERGENCE;
        hi *= 2.0;
    }

    double y = bond->coupon_rate;
    if (!(y > lo && y < hi)) y = 0.5 * (lo + hi);
    for (int iter = 0; iter < 200; ++iter) {
        const pv_sums s = discount(bond, &pos, y, compounding);
        const double diff = s.pv - target;
        if (fabs(diff) <= 8.0 * DBL_EPSILON * target) {
            *yield_out = y;
            return QSL_OK;
        }
        if (diff > 0.0) lo = y; /* price too high -> yield must rise */
        else hi = y;
        const double growth = compounding == QSL_COMPOUNDED ? 1.0 + y / f : 1.0;
        const double dprice = -s.pv_t / growth;
        double next = dprice < 0.0 ? y - diff / dprice : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
        if (fabs(next - y) <= 4.0 * DBL_EPSILON * fmax(1.0, fabs(y))) {
            *yield_out = next;
            return QSL_OK;
        }
        y = next;
    }
    return QSL_E_NO_CONVERGENCE;
}
