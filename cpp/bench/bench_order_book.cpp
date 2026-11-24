// Order book benchmark.
//
// Pre-generates a realistic message stream (passive adds near the touch,
// cancels, marketable orders, cancel/replaces), then replays it into the
// engineered OrderBook and into the std::map ReferenceBook baseline. Reports
// throughput, the per-message latency distribution, and checks that both
// engines produced identical executions.
//
//   bench_lob [messages=5000000] [seed=42]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "order_flow.hpp"
#include "qsl/lob/reference_book.hpp"

using namespace qsl::lob;
using Clock = std::chrono::steady_clock;

namespace {

using namespace qsl::bench;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 5'000'000;
    const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 42;

    std::size_t peak = 0;
    const std::vector<Message> msgs = generate(n, seed, peak);
    std::size_t counts[4] = {};
    for (const Message& m : msgs) ++counts[static_cast<int>(m.kind)];
    std::printf("Order book benchmark: %zu messages (add %.0f%%, cancel %.0f%%, replace %.0f%%, market %.0f%%)\n",
                n, 100.0 * static_cast<double>(counts[0]) / static_cast<double>(n),
                100.0 * static_cast<double>(counts[1]) / static_cast<double>(n),
                100.0 * static_cast<double>(counts[2]) / static_cast<double>(n),
                100.0 * static_cast<double>(counts[3]) / static_cast<double>(n));
    std::printf("  ladder %lld ticks, peak resting orders %zu, seed %llu\n\n",
                static_cast<long long>(kConfig.max_price - kConfig.min_price + 1), peak,
                static_cast<unsigned long long>(seed));

    // Throughput: best of 5 full replays into a fresh book.
    double best_fast = 1e300;
    ExecStats fast_stats;
    for (int rep = 0; rep < 5; ++rep) {
        auto book = std::make_unique<OrderBook<StatsListener>>(kConfig);
        const auto t0 = Clock::now();
        for (const Message& m : msgs) apply(*book, m);
        best_fast = std::min(best_fast, seconds_since(t0));
        fast_stats = book->listener().stats;
    }

    // Baseline: the std::map / std::list reference book (one replay).
    ExecStats ref_stats;
    double ref_time;
    {
        ReferenceBook ref(kConfig);
        const auto t0 = Clock::now();
        for (const Message& m : msgs) {
            apply(ref, m);
            for (const Trade& t : ref.trades()) ref_stats.add(t);
            ref.clear_trades();
        }
        ref_time = seconds_since(t0);
    }

    const double dn = static_cast<double>(n);
    std::printf("  %-28s %10s %10s\n", "engine", "msgs/s", "ns/msg");
    std::printf("  %-28s %9.1fM %10.1f\n", "qsl::lob::OrderBook", dn / best_fast / 1e6, best_fast / dn * 1e9);
    std::printf("  %-28s %9.1fM %10.1f   (%.1fx slower)\n\n", "std::map ReferenceBook", dn / ref_time / 1e6,
                ref_time / dn * 1e9, ref_time / best_fast);

    // Per-message latency distribution (each message timed individually).
    std::vector<std::uint32_t> lat(n);
    {
        auto book = std::make_unique<OrderBook<StatsListener>>(kConfig);
        for (std::size_t i = 0; i < n; ++i) {
            const auto t0 = Clock::now();
            apply(*book, msgs[i]);
            const auto t1 = Clock::now();
            lat[i] = static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
    }
    // Timer granularity: smallest non-zero step of steady_clock (41.7 ns on
    // Apple Silicon's 24 MHz timebase, ~1 ns on x86 with an invariant TSC).
    std::int64_t resolution = 1'000'000;
    for (int trial = 0; trial < 2000; ++trial) {
        const auto t0 = Clock::now();
        auto t1 = Clock::now();
        while (t1 == t0) t1 = Clock::now();
        resolution = std::min<std::int64_t>(resolution, std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    std::sort(lat.begin(), lat.end());
    auto pct = [&](double p) {
        return lat[std::min(n - 1, static_cast<std::size_t>(p / 100.0 * static_cast<double>(n)))];
    };
    std::printf("  per-message latency in ns (each message timed alone; clock resolution %lld ns):\n",
                static_cast<long long>(resolution));
    std::printf("    p50 %u   p90 %u   p99 %u   p99.9 %u   p99.99 %u   max %u\n\n", pct(50), pct(90), pct(99),
                pct(99.9), pct(99.99), lat.back());

    const bool identical = fast_stats == ref_stats;
    std::printf("  executions identical across engines: %s (%llu trades, %llu shares)\n", identical ? "yes" : "NO",
                static_cast<unsigned long long>(fast_stats.trades),
                static_cast<unsigned long long>(fast_stats.volume));
    return identical ? 0 : 1;
}
