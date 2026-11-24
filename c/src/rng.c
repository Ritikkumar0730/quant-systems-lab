/* xoshiro256** generator (Blackman & Vigna, 2018) with SplitMix64 seeding. */
#include "internal.h"

#include <math.h>

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void qsl_rng_seed(qsl_rng *rng, uint64_t seed) {
    /* SplitMix64 expands one word into a well-mixed, never all-zero state. */
    uint64_t sm = seed;
    for (int i = 0; i < 4; ++i) rng->s[i] = splitmix64(&sm);
    rng->spare = 0.0;
    rng->has_spare = 0;
}

uint64_t qsl_rng_next(qsl_rng *rng) {
    uint64_t *s = rng->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

double qsl_rng_uniform(qsl_rng *rng) {
    /* Top 53 bits -> a double in [0, 1) on a uniform 2^-53 grid. */
    return (double)(qsl_rng_next(rng) >> 11) * 0x1.0p-53;
}

double qsl_rng_normal(qsl_rng *rng) {
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }
    /* Marsaglia polar method: accept (u, v) uniformly in the unit disc. */
    double u;
    double v;
    double s;
    do {
        u = 2.0 * qsl_rng_uniform(rng) - 1.0;
        v = 2.0 * qsl_rng_uniform(rng) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double m = sqrt(-2.0 * log(s) / s);
    rng->spare = v * m;
    rng->has_spare = 1;
    return u * m;
}
