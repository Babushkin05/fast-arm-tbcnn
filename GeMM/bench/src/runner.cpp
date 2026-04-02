#include "runner.hpp"
#include <thread>

namespace bench {

void sleep_ms(std::uint32_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

double calculate_gflops(std::uint32_t m, std::uint32_t n, std::uint32_t k, double time_ms) {
    // Total operations: m * k * (2 * n) multiplications and additions
    const std::uint64_t ops = static_cast<std::uint64_t>(m) * n * k * 2;
    const double seconds = time_ms / 1000.0;
    return static_cast<double>(ops) / seconds / 1e9;
}

} // namespace bench
