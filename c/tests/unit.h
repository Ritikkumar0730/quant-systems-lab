/* Minimal assertion harness for the C test suite (no external dependency). */
#ifndef QSL_UNIT_H
#define QSL_UNIT_H

#include <math.h>
#include <stdio.h>

static int unit_failures = 0;
static int unit_checks = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        ++unit_checks;                                                             \
        if (!(cond)) {                                                             \
            ++unit_failures;                                                       \
            fprintf(stderr, "\n  %s:%d: CHECK(%s) failed", __FILE__, __LINE__, #cond); \
        }                                                                          \
    } while (0)

/* |actual - expected| <= tol, reporting both values on failure. */
#define CHECK_CLOSE(actual, expected, tol)                                         \
    do {                                                                           \
        ++unit_checks;                                                             \
        const double a_ = (actual);                                                \
        const double e_ = (expected);                                              \
        const double t_ = (tol);                                                   \
        if (!(fabs(a_ - e_) <= t_)) {                                              \
            ++unit_failures;                                                       \
            fprintf(stderr, "\n  %s:%d: %s = %.17g, expected %.17g (|diff| %.3g > %.3g)", \
                    __FILE__, __LINE__, #actual, a_, e_, fabs(a_ - e_), t_);       \
        }                                                                          \
    } while (0)

/* Mixed absolute/relative tolerance: |a - e| <= tol * max(1, |e|). */
#define CHECK_NEAR(actual, expected, tol)                                          \
    CHECK_CLOSE(actual, expected, (tol) * fmax(1.0, fabs(expected)))

#define RUN_TEST(fn)                                                               \
    do {                                                                           \
        const int before_ = unit_failures;                                         \
        printf("  %-44s", #fn);                                                    \
        fflush(stdout);                                                            \
        fn();                                                                      \
        printf("%s\n", unit_failures == before_ ? "ok" : " FAILED");               \
    } while (0)

#define UNIT_SUMMARY()                                                             \
    (printf("\n%d checks, %d failures\n", unit_checks, unit_failures), unit_failures == 0 ? 0 : 1)

#endif /* QSL_UNIT_H */
