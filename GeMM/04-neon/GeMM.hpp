#include <cstdint>
#include <memory>
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
// C is ternary
// n mod 64 == 0
// m mod 8 == 0
// k mod 8 == 0
// A: pair(Aplus, Aminus) where each is a unique_ptr to a byte-array of size (m * (n/8)).
// B: unique_ptr to byte-array of size (k * (n/8)) (column-major bitplanes as before).
// Returns: pair(Cplus, Cminus) as unique_ptr<std::uint8_t[]> each of size (m * (k/8)).
std::pair<std::unique_ptr<std::uint8_t[]>, std::unique_ptr<std::uint8_t[]>> GemmTBN_Blocked(
  std::pair<std::unique_ptr<const std::uint8_t[]>, std::unique_ptr<const std::uint8_t[]>> A,
  std::unique_ptr<const std::uint8_t[]> B,
  std::uint32_t m, std::uint32_t n, std::uint32_t k,
  const TilingParams& tp);

// Unpack ternary planes into an int8_t matrix managed by unique_ptr
std::unique_ptr<int8_t[]> UnpackTernaryRowMajor(const std::uint8_t* Cplus, const std::uint8_t* Cminus,
                        std::uint32_t m, std::uint32_t k);
