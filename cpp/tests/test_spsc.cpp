// SPSC queue tests: single-thread semantics and a two-thread stress test.
// Run under ThreadSanitizer with `make tsan` to prove the absence of races.
#include <array>
#include <cstdint>
#include <memory>
#include <thread>

#include "qsl/concurrency/spsc_queue.hpp"
#include "testing.hpp"

namespace {

/// Payload with redundancy so a torn or stale read is detectable.
struct Message {
    std::uint64_t seq;
    std::uint64_t square;
    std::uint64_t check;
};

constexpr Message make_message(std::uint64_t i) { return {i, i * i, ~(i ^ 0x5bd1e995ULL)}; }

}  // namespace

TEST_CASE(spsc_fifo_and_capacity) {
    auto q = std::make_unique<qsl::SpscQueue<int, 8>>();
    int out = 0;
    CHECK(!q->try_pop(out));
    for (int i = 0; i < 8; ++i) CHECK(q->try_push(i));
    CHECK(!q->try_push(99));  // full
    CHECK(q->size() == 8);
    for (int i = 0; i < 8; ++i) {
        CHECK(q->try_pop(out));
        CHECK(out == i);
    }
    CHECK(!q->try_pop(out));
}

TEST_CASE(spsc_wraparound) {
    auto q = std::make_unique<qsl::SpscQueue<std::uint64_t, 4>>();
    std::uint64_t next_in = 0, next_out = 0, out = 0;
    for (int round = 0; round < 10000; ++round) {
        const int burst = 1 + round % 4;
        for (int i = 0; i < burst; ++i) CHECK(q->try_push(next_in++));
        for (int i = 0; i < burst; ++i) {
            CHECK(q->try_pop(out));
            CHECK(out == next_out++);
        }
    }
    CHECK(q->size() == 0);
}

TEST_CASE(spsc_two_thread_stress) {
    constexpr std::uint64_t kCount = 2'000'000;
    auto q = std::make_unique<qsl::SpscQueue<Message, 1024>>();
    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kCount; ++i) {
            const Message m = make_message(i);
            while (!q->try_push(m)) std::this_thread::yield();
        }
    });
    std::uint64_t expected = 0;
    bool ordered = true, intact = true;
    Message m{};
    while (expected < kCount) {
        if (!q->try_pop(m)) {
            std::this_thread::yield();
            continue;
        }
        ordered &= m.seq == expected;
        const Message ref = make_message(m.seq);
        intact &= m.square == ref.square && m.check == ref.check;
        ++expected;
    }
    producer.join();
    CHECK(ordered);
    CHECK(intact);
    CHECK(!q->try_pop(m));
}

TEST_CASE(spsc_bulk_operations) {
    auto q = std::make_unique<qsl::SpscQueue<int, 8>>();
    const std::array<int, 5> a{1, 2, 3, 4, 5};
    CHECK(q->push_bulk(a) == 5);
    CHECK(q->push_bulk(a) == 3);  // only 3 slots left
    std::array<int, 16> out{};
    CHECK(q->pop_bulk(std::span<int>(out.data(), 6)) == 6);
    CHECK(out[0] == 1 && out[4] == 5 && out[5] == 1);
    CHECK(q->pop_bulk(out) == 2 && out[0] == 2 && out[1] == 3);
    CHECK(q->pop_bulk(out) == 0);
    CHECK(q->push_bulk(std::span<const int>{}) == 0);
}

TEST_CASE(spsc_bulk_two_thread_stress) {
    // Producer alternates bulk and single pushes, consumer pops in bulk.
    constexpr std::uint64_t kCount = 2'000'000;
    auto q = std::make_unique<qsl::SpscQueue<Message, 256>>();
    std::thread producer([&] {
        std::array<Message, 37> batch{};
        std::uint64_t i = 0;
        while (i < kCount) {
            if (i % 3 == 0) {
                while (!q->try_push(make_message(i))) std::this_thread::yield();
                ++i;
                continue;
            }
            const std::size_t want = std::min<std::uint64_t>(batch.size(), kCount - i);
            for (std::size_t k = 0; k < want; ++k) batch[k] = make_message(i + k);
            std::size_t sent = 0;
            while (sent < want) {
                sent += q->push_bulk(std::span<const Message>(batch.data() + sent, want - sent));
                if (sent < want) std::this_thread::yield();
            }
            i += want;
        }
    });
    std::array<Message, 64> out{};
    std::uint64_t expected = 0;
    bool ok = true;
    while (expected < kCount) {
        const std::size_t n = q->pop_bulk(out);
        if (n == 0) std::this_thread::yield();
        for (std::size_t k = 0; k < n; ++k) {
            const Message ref = make_message(expected++);
            ok &= out[k].seq == ref.seq && out[k].square == ref.square && out[k].check == ref.check;
        }
    }
    producer.join();
    CHECK(ok);
    CHECK(expected == kCount);
}
