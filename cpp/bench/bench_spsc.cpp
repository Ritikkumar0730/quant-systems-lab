// SPSC queue benchmarks:
//   1. raw throughput between two threads,
//   2. round-trip latency (ping-pong over two queues),
//   3. a two-thread pipeline: gateway thread -> SpscQueue -> matching thread
//      running the OrderBook, compared with the same flow on one thread.
//
//   bench_spsc [messages=20000000]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "order_flow.hpp"
#include "qsl/concurrency/spsc_queue.hpp"

using Clock = std::chrono::steady_clock;
using qsl::SpscQueue;

namespace {

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

/// Producer pushes one element at a time (batch == 1) or in batches.
double throughput(std::uint64_t n, std::size_t batch) {
    auto q = std::make_unique<SpscQueue<std::uint64_t, 1u << 16>>();
    std::uint64_t checksum = 0;
    const auto t0 = Clock::now();
    std::thread consumer([&] {
        std::uint64_t buf[256];
        for (std::uint64_t got = 0; got < n;) {
            const std::size_t k = batch == 1 ? (q->try_pop(buf[0]) ? 1 : 0) : q->pop_bulk(std::span<std::uint64_t>(buf, 256));
            for (std::size_t j = 0; j < k; ++j) checksum += buf[j];
            got += k;
        }
    });
    std::vector<std::uint64_t> chunk(batch);
    for (std::uint64_t i = 0; i < n;) {
        if (batch == 1) {
            while (!q->try_push(i)) {
            }
            ++i;
            continue;
        }
        const std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(batch, n - i));
        for (std::size_t j = 0; j < want; ++j) chunk[j] = i + j;
        for (std::size_t sent = 0; sent < want;)
            sent += q->push_bulk(std::span<const std::uint64_t>(chunk.data() + sent, want - sent));
        i += want;
    }
    consumer.join();
    const double secs = seconds_since(t0);
    if (checksum != n * (n - 1) / 2) std::printf("  checksum mismatch!\n");
    return static_cast<double>(n) / secs;
}

std::vector<std::uint32_t> round_trips(std::size_t n) {
    auto ping = std::make_unique<SpscQueue<std::uint64_t, 1024>>();
    auto pong = std::make_unique<SpscQueue<std::uint64_t, 1024>>();
    std::thread echo([&] {
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < n; ++i) {
            while (!ping->try_pop(v)) {
            }
            while (!pong->try_push(v)) {
            }
        }
    });
    std::vector<std::uint32_t> rtt(n);
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto t0 = Clock::now();
        while (!ping->try_push(i)) {
        }
        while (!pong->try_pop(v)) {
        }
        rtt[i] = static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
    }
    echo.join();
    std::sort(rtt.begin(), rtt.end());
    return rtt;
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20'000'000;
    std::printf("SPSC queue benchmark (%u hardware threads)\n\n", std::thread::hardware_concurrency());

    for (const std::size_t batch : {std::size_t{1}, std::size_t{16}, std::size_t{64}}) {
        double best = 0;
        for (int rep = 0; rep < 3; ++rep) best = std::max(best, throughput(n, batch));
        std::printf("  throughput, batch %2zu: %7.1f M msgs/s (%5.2f ns/msg, best of 3, %llu x uint64)\n", batch,
                    best / 1e6, 1e9 / best, static_cast<unsigned long long>(n));
    }

    const auto rtt = round_trips(1'000'000);
    auto pct = [&](double p) { return rtt[std::min(rtt.size() - 1, static_cast<std::size_t>(p / 100.0 * static_cast<double>(rtt.size())))]; };
    std::printf("  round trip (ping-pong, ns): p50 %u   p90 %u   p99 %u   p99.9 %u   (one-way ~ p50/2 = %u)\n\n",
                pct(50), pct(90), pct(99), pct(99.9), pct(50) / 2);

    // Pipeline: the same realistic order flow as bench_lob, handed from a
    // gateway thread to a matching thread through the queue.
    using namespace qsl::bench;
    std::size_t peak = 0;
    const std::vector<Message> msgs = generate(5'000'000, 7, peak);

    auto single = std::make_unique<OrderBook<StatsListener>>(kConfig);
    auto t0 = Clock::now();
    for (const Message& m : msgs) apply(*single, m);
    const double single_secs = seconds_since(t0);

    auto book = std::make_unique<OrderBook<StatsListener>>(kConfig);
    auto queue = std::make_unique<SpscQueue<Message, 4096>>();
    t0 = Clock::now();
    std::thread matcher([&] {  // drains in bulk: one acquire per burst, not per message
        Message burst[64];
        for (std::size_t done = 0; done < msgs.size();) {
            const std::size_t k = queue->pop_bulk(std::span<Message>(burst, 64));
            for (std::size_t j = 0; j < k; ++j) apply(*book, burst[j]);
            done += k;
        }
    });
    for (const Message& m : msgs)
        while (!queue->try_push(m)) {
        }
    matcher.join();
    const double pipe_secs = seconds_since(t0);

    const double dn = static_cast<double>(msgs.size());
    std::printf("  order flow, %zu msgs (%zu-byte messages):\n", msgs.size(), sizeof(Message));
    std::printf("    single thread                %6.1f M msgs/s\n", dn / single_secs / 1e6);
    std::printf("    gateway -> SPSC -> matcher   %6.1f M msgs/s  (matcher drains in bursts of up to 64)\n",
                dn / pipe_secs / 1e6);
    const bool same = single->listener().stats == book->listener().stats;
    std::printf("    identical executions: %s\n", same ? "yes" : "NO");
    return same ? 0 : 1;
}
