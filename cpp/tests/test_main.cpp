// Test runner: `test_cpp [substring]` runs every test whose name contains it.
#include <chrono>
#include <cstdio>
#include <string_view>

#include "testing.hpp"

int main(int argc, char** argv) {
    const std::string_view filter = argc > 1 ? argv[1] : "";
    if (filter.empty()) std::printf("C++ tests\n");
    else std::printf("C++ tests matching '%s'\n", argv[1]);

    int ran = 0;
    for (const auto& test : qsl::testing::registry()) {
        if (!filter.empty() && std::string_view(test.name).find(filter) == std::string_view::npos) continue;
        const long before = qsl::testing::failure_count();
        std::printf("  %-58s", test.name);
        std::fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();
        test.fn();
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
        std::printf("%s (%.0f ms)\n", qsl::testing::failure_count() == before ? "ok" : " FAILED", ms);
        ++ran;
    }
    std::printf("\n%d tests, %ld checks, %ld failures\n", ran, qsl::testing::check_count(),
                qsl::testing::failure_count());
    return qsl::testing::failure_count() == 0 && ran > 0 ? 0 : 1;
}
