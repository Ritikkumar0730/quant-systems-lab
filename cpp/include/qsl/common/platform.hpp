#pragma once

#include <cstddef>

namespace qsl {

/// Padding unit for data touched by different threads. Apple M-series cores
/// use 128-byte lines and x86 spatial prefetchers fetch 64-byte lines in
/// pairs, so 128 bytes avoids false sharing on both (folly uses the same).
inline constexpr std::size_t kCacheLineSize = 128;

}  // namespace qsl
