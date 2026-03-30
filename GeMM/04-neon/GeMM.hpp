#include <cstdint>
#include <utility>

// Pipeline parameters (tunable for cache sizes)
// Example defaults for AArch64 mobile CPUs (adjust after profiling)
struct TilingParams {
    std::uint32_t mblk; // outer block on rows (L2)
    std::uint32_t nblk; // outer block on cols (L2)
    std::uint32_t kblk; // outer block on depth (L2)
    std::uint32_t mmk;  // microkernel rows (L1)
    std::uint32_t nmk;  // microkernel cols (L1)
};

// A - m x n, B - n x k, tp parameters of cache; result - m x k
// A is ternary: (0, 1) -> -1; (0, 0) -> 0; (1, 0) -> 1
// B is binary: 0 -> 1; 1 -> -1
// C is int32 (accumulated diff values)
// n mod 128 == 0 (for NEON 128-bit alignment)
// kblk mod 128 == 0 (for NEON 128-bit alignment)
// m mod mmk == 0
// k mod nmk == 0
std::int32_t* GemmTBN_Blocked(
    const std::uint8_t* Ap, const std::uint8_t* Am, const std::uint8_t* B,
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    const TilingParams& tp
);
