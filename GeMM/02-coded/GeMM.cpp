#include <cstdint>
#include <utility>
#include <cstring>
#include <stdexcept>
#include <bit>

static inline std::uint64_t load_u64(const std::uint8_t* p) {
    std::uint64_t v;
    std::memcpy(&v, p, sizeof(std::uint64_t));
    return v;
}

// A - m x n, B - n x k; result - m x k
// A is ternary: (0, 1) -> -1; (0, 0) -> 0; (1, 0) -> 1
// B is binary: 0 -> 1; 1 -> -1
// C is ternary
// n mod 64 == 0
// m mod 8 == 0
// k mod 8 == 0
std::pair<std::uint8_t*, std::uint8_t*> GeMMCoded(std::uint8_t * Ap, std::uint8_t * Am, std::uint8_t * B, std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    if ((m % 8) != 0 || (n % 64) != 0 || (k % 8) != 0) {
        throw std::invalid_argument("GeMM: m and k must be multiples of 8; n must be multiple of 64");
    }
    if (Ap == nullptr || Am == nullptr || B == nullptr) {
        throw std::invalid_argument("GeMM: null input buffer");
    }

    const std::uint32_t rowBytesA = n / 8;
    const std::uint32_t colBytesB = n / 8;
    const std::uint32_t rowBytesC = k / 8;

    std::uint8_t* Cplus  = new std::uint8_t[m * rowBytesC];
    std::uint8_t* Cminus = new std::uint8_t[m * rowBytesC];
    std::memset(Cplus,  0, m * rowBytesC);
    std::memset(Cminus, 0, m * rowBytesC);

    // arm popcount works efficiently for 64-bit integers
    const std::uint32_t chunks64 = rowBytesA / 8;

    for (std::uint32_t i = 0; i < m; ++i) {
        const std::uint8_t* ArowP = Ap + i * rowBytesA;
        const std::uint8_t* ArowM = Am + i * rowBytesA;

        std::uint8_t* CrowP = Cplus  + i * rowBytesC;
        std::uint8_t* CrowM = Cminus + i * rowBytesC;

        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint8_t* Bcol = B + j * colBytesB;

            int posCount = 0;
            int negCount = 0;

            // 64-bit chunks only (n multiple of 64)
            for (std::uint32_t c = 0; c < chunks64; ++c) {
                const std::uint8_t* aptr = ArowP + c * 8;
                const std::uint8_t* amptr = ArowM + c * 8;
                const std::uint8_t* bptr  = Bcol  + c * 8;

                const std::uint64_t ap = load_u64(aptr);
                const std::uint64_t am = load_u64(amptr);
                const std::uint64_t bc = load_u64(bptr);

                const std::uint64_t posMask = (ap | bc) & (am | ~bc);
                const std::uint64_t negMask = (ap | ~bc) & (am | bc);

                posCount += std::popcount(posMask);
                negCount += std::popcount(negMask);
            }

            const int diff = posCount - negCount;
            const std::uint32_t byteIdx = j / 8;
            const std::uint8_t  bitMask = static_cast<std::uint8_t>(1u << (j & 7));

            if (diff > 0) {
                CrowP[byteIdx] |= bitMask; // +1 
            } else if (diff < 0) {
                CrowM[byteIdx] |= bitMask; // -1 
            }
        }
    }

    auto result = std::make_pair(Cplus, Cminus);
    return result;
}
