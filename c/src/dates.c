/* Civil-calendar arithmetic and day count conventions. */
#include "internal.h"

#include <math.h>

int qsl_is_leap_year(int32_t year) {
    return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

int32_t qsl_days_in_month(int32_t year, int32_t month) {
    static const int32_t days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) return 0;
    return month == 2 && qsl_is_leap_year(year) ? 29 : days[month - 1];
}

int qsl_date_is_valid(qsl_date d) {
    return d.month >= 1 && d.month <= 12 && d.day >= 1 &&
           d.day <= qsl_days_in_month(d.year, d.month);
}

/* Howard Hinnant's days_from_civil: exact for the whole int32 year range. */
int64_t qsl_date_to_serial(qsl_date d) {
    const int64_t y = (int64_t)d.year - (d.month <= 2 ? 1 : 0);
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;                                      /* [0, 399]    */
    const int64_t m = d.month;
    const int64_t doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d.day - 1; /* [0, 365]    */
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;              /* [0, 146096] */
    return era * 146097 + doe - 719468;
}

qsl_date qsl_date_from_serial(int64_t serial) {
    const int64_t z = serial + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;                                         /* [0, 146096] */
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;    /* [0, 399]    */
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                  /* [0, 365]    */
    const int64_t mp = (5 * doy + 2) / 153;                                       /* [0, 11]     */
    const int64_t day = doy - (153 * mp + 2) / 5 + 1;                             /* [1, 31]     */
    const int64_t month = mp < 10 ? mp + 3 : mp - 9;                              /* [1, 12]     */
    const int64_t year = yoe + era * 400 + (month <= 2 ? 1 : 0);
    return (qsl_date){(int32_t)year, (int32_t)month, (int32_t)day};
}

static int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

qsl_date qsl_date_add_months(qsl_date d, int32_t months, int end_of_month) {
    const int64_t index = (int64_t)d.year * 12 + (d.month - 1) + months;
    const int64_t year = floor_div(index, 12);
    const int32_t y = (int32_t)year;
    const int32_t m = (int32_t)(index - year * 12) + 1;
    const int32_t dim = qsl_days_in_month(y, m);
    int32_t day = d.day;
    if (end_of_month && d.day == qsl_days_in_month(d.year, d.month)) day = dim;
    else if (day > dim) day = dim;
    return (qsl_date){y, m, day};
}

static int64_t actual_days(qsl_date d1, qsl_date d2) {
    return qsl_date_to_serial(d2) - qsl_date_to_serial(d1);
}

/* 30/360 family: 360 (Y2 - Y1) + 30 (M2 - M1) + (D2 - D1) after day rules. */
static double thirty_360(qsl_date d1, qsl_date d2, int european) {
    int32_t dd1 = d1.day;
    int32_t dd2 = d2.day;
    if (european) {
        if (dd1 == 31) dd1 = 30;
        if (dd2 == 31) dd2 = 30;
    } else { /* ISDA bond basis */
        if (dd1 == 31) dd1 = 30;
        if (dd2 == 31 && dd1 == 30) dd2 = 30;
    }
    const int64_t days =
        360 * ((int64_t)d2.year - d1.year) + 30 * ((int64_t)d2.month - d1.month) + (dd2 - dd1);
    return (double)days / 360.0;
}

static double act_act_isda(qsl_date d1, qsl_date d2) {
    if (qsl_date_to_serial(d2) < qsl_date_to_serial(d1)) return -act_act_isda(d2, d1);
    if (d1.year == d2.year)
        return (double)actual_days(d1, d2) / (qsl_is_leap_year(d1.year) ? 366.0 : 365.0);
    const qsl_date end_y1 = {d1.year + 1, 1, 1};
    const qsl_date start_y2 = {d2.year, 1, 1};
    return (double)actual_days(d1, end_y1) / (qsl_is_leap_year(d1.year) ? 366.0 : 365.0) +
           (double)(d2.year - d1.year - 1) +
           (double)actual_days(start_y2, d2) / (qsl_is_leap_year(d2.year) ? 366.0 : 365.0);
}

double qsl_year_fraction(qsl_day_count dc, qsl_date d1, qsl_date d2) {
    if (!qsl_date_is_valid(d1) || !qsl_date_is_valid(d2)) return NAN;
    switch (dc) {
    case QSL_DC_ACT_360: return (double)actual_days(d1, d2) / 360.0;
    case QSL_DC_ACT_365F: return (double)actual_days(d1, d2) / 365.0;
    case QSL_DC_30_360_BOND_BASIS: return thirty_360(d1, d2, 0);
    case QSL_DC_30E_360: return thirty_360(d1, d2, 1);
    case QSL_DC_ACT_ACT_ISDA: return act_act_isda(d1, d2);
    case QSL_DC_ACT_ACT_ICMA: return NAN; /* needs a reference period */
    }
    return NAN;
}

double qsl_year_fraction_icma(qsl_date d1, qsl_date d2, qsl_date ref_start,
                              qsl_date ref_end, int frequency) {
    if (frequency <= 0 || !qsl_date_is_valid(d1) || !qsl_date_is_valid(d2) ||
        !qsl_date_is_valid(ref_start) || !qsl_date_is_valid(ref_end))
        return NAN;
    const int64_t period = actual_days(ref_start, ref_end);
    if (period <= 0) return NAN;
    return (double)actual_days(d1, d2) / ((double)frequency * (double)period);
}
