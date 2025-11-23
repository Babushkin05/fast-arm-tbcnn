#include <cstdint>
#include <utility>
#include <cstring>
#include <stdexcept>
#include <bit>

static inline uint64_t load_u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(uint64_t));
    return v;
}

// A - m x n, B - n x k; result - m x k
// A is ternary: (0, 1) -> -1; (0, 0) -> 0; (1, 0) -> 1
// B is binary: 0 -> 1; 1 -> -1
// C is ternary
// n mod 64 == 0
// m mod 8 == 0
// k mod 8 == 0
std::pair<uint8_t*, uint8_t*> GeMMCoded(uint8_t * Ap, uint8_t * Am, uint8_t * B, uint32_t m, uint32_t n, uint32_t k) {
    if ((m % 8) != 0 || (n % 64) != 0 || (k % 8) != 0) {
        throw std::invalid_argument("GeMM: m and k must be multiples of 8; n must be multiple of 64");
    }
    if (Ap == nullptr || Am == nullptr || B == nullptr) {
        throw std::invalid_argument("GeMM: null input buffer");
    }

    const uint32_t rowBytesA = n / 8;
    const uint32_t colBytesB = n / 8;
    const uint32_t rowBytesC = k / 8;

    uint8_t* Cplus  = new uint8_t[m * rowBytesC];
    uint8_t* Cminus = new uint8_t[m * rowBytesC];
    std::memset(Cplus,  0, m * rowBytesC);
    std::memset(Cminus, 0, m * rowBytesC);

    // arm popcount works efficiently for 64-bit integers
    const uint32_t chunks64 = rowBytesA / 8;

    for (uint32_t i = 0; i < m; ++i) {
        const uint8_t* ArowP = Ap + i * rowBytesA;
        const uint8_t* ArowM = Am + i * rowBytesA;

        uint8_t* CrowP = Cplus  + i * rowBytesC;
        uint8_t* CrowM = Cminus + i * rowBytesC;

        for (uint32_t j = 0; j < k; ++j) {
            const uint8_t* Bcol = B + j * colBytesB;

            int posCount = 0;
            int negCount = 0;

            // 64-bit chunks only (n multiple of 64)
            for (uint32_t c = 0; c < chunks64; ++c) {
                const uint8_t* aptr = ArowP + c * 8;
                const uint8_t* amptr = ArowM + c * 8;
                const uint8_t* bptr  = Bcol  + c * 8;

                const uint64_t ap = load_u64(aptr);
                const uint64_t am = load_u64(amptr);
                const uint64_t bc = load_u64(bptr);

                const uint64_t posMask = (ap | bc) & (am | ~bc);
                const uint64_t negMask = (ap | ~bc) & (am | bc);

                posCount += std::popcount(posMask);
                negCount += std::popcount(negMask);
            }

            const int diff = posCount - negCount;
            const uint32_t byteIdx = j / 8;
            const uint8_t  bitMask = static_cast<uint8_t>(1u << (j & 7));

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