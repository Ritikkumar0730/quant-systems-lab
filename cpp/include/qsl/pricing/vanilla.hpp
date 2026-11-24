#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qsl::pricing {

enum class OptionType { Call, Put };
enum class Exercise { European, American };

/// Vanilla option on an asset with continuous dividend yield (or foreign
/// rate) `dividend`, flat risk-free `rate` and flat volatility `vol`.
struct Vanilla {
    OptionType type = OptionType::Call;
    Exercise exercise = Exercise::European;
    double spot = 100.0;
    double strike = 100.0;
    double maturity = 1.0;  ///< years
    double rate = 0.0;
    double dividend = 0.0;
    double vol = 0.2;
};

[[nodiscard]] inline double payoff(OptionType type, double spot, double strike) noexcept {
    return type == OptionType::Call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
}

inline void validate(const Vanilla& o) {
    if (!(o.spot > 0.0) || !(o.strike > 0.0) || !(o.maturity > 0.0) || !(o.vol > 0.0) ||
        !std::isfinite(o.rate) || !std::isfinite(o.dividend) || !std::isfinite(o.spot) ||
        !std::isfinite(o.strike) || !std::isfinite(o.maturity) || !std::isfinite(o.vol))
        throw std::invalid_argument("vanilla option: spot, strike, maturity and vol must be positive and finite");
}

}  // namespace qsl::pricing
