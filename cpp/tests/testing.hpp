#pragma once
// Minimal self-registering test framework (no third-party dependency).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace qsl::testing {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}
inline long& failure_count() {
    static long failures = 0;
    return failures;
}
inline long& check_count() {
    static long checks = 0;
    return checks;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void fail(const char* file, int line, const std::string& message) {
    ++failure_count();
    std::fprintf(stderr, "\n  %s:%d: %s", file, line, message.c_str());
}

/// SplitMix64: tiny, fast and identical on every platform, so randomized
/// tests replay the same sequences everywhere (std:: distributions do not).
struct Rng {
    std::uint64_t state;
    std::uint64_t next() noexcept {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    std::uint64_t below(std::uint64_t n) noexcept { return next() % n; }
    double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    bool chance(double p) noexcept { return uniform() < p; }
};

inline std::string describe_close(const char* expr, double actual, double expected, double tol) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s = %.17g, expected %.17g (|diff| %.3g > %.3g)", expr, actual,
                  expected, std::fabs(actual - expected), tol);
    return buf;
}

}  // namespace qsl::testing

#define QSL_TEST_CONCAT_(a, b) a##b
#define QSL_TEST_CONCAT(a, b) QSL_TEST_CONCAT_(a, b)

#define TEST_CASE(name)                                                                       \
    static void name();                                                                       \
    static const ::qsl::testing::Registrar QSL_TEST_CONCAT(qsl_registrar_, name){#name, &name}; \
    static void name()

#define CHECK(cond)                                                                \
    do {                                                                           \
        ++::qsl::testing::check_count();                                           \
        if (!(cond)) ::qsl::testing::fail(__FILE__, __LINE__, "CHECK(" #cond ") failed"); \
    } while (0)

/// Stops the current test on failure (use when later checks would be noise).
#define REQUIRE(cond)                                                              \
    do {                                                                           \
        ++::qsl::testing::check_count();                                           \
        if (!(cond)) {                                                             \
            ::qsl::testing::fail(__FILE__, __LINE__, "REQUIRE(" #cond ") failed"); \
            return;                                                                \
        }                                                                          \
    } while (0)

#define CHECK_CLOSE(actual, expected, tol)                                                     \
    do {                                                                                       \
        ++::qsl::testing::check_count();                                                       \
        const double qa_ = static_cast<double>(actual);                                        \
        const double qe_ = static_cast<double>(expected);                                      \
        const double qt_ = static_cast<double>(tol);                                           \
        if (!(std::fabs(qa_ - qe_) <= qt_))                                                    \
            ::qsl::testing::fail(__FILE__, __LINE__,                                           \
                                 ::qsl::testing::describe_close(#actual, qa_, qe_, qt_));      \
    } while (0)

/// |actual - expected| <= tol * max(1, |expected|)
#define CHECK_NEAR(actual, expected, tol) \
    CHECK_CLOSE(actual, expected, (tol) * std::fmax(1.0, std::fabs(static_cast<double>(expected))))
