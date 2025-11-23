#include <cstdint>
#include <utility>


// Pipeline parameters (tunable for cache sizes)
// Example defaults for AArch64 mobile CPUs (adjust after profiling)
struct TilingParams {
    uint32_t mblk; // outer block on rows (L2)
    uint32_t nblk; // outer block on cols (L2)
    uint32_t kblk; // outer block on depth (L2)
    uint32_t mmk;  // microkernel rows (L1)
    uint32_t nmk;  // microkernel cols (L1)
};


// A - m x n, B - n x k, tp parameters of cache; result - m x k
// A is ternary: (0, 1) -> -1; (0, 0) -> 0; (1, 0) -> 1
// B is binary: 0 -> 1; 1 -> -1
// C is ternary
// n mod 64 == 0
// m mod 8 == 0
// k mod 8 == 0
std::pair<uint8_t*, uint8_t*> GemmTBN_Blocked(
    const uint8_t* Ap, const uint8_t* Am, const uint8_t* B,
    uint32_t m, uint32_t n, uint32_t k,
    const TilingParams& tp
);