#include <cstdint>
#include <utility>

std::pair<uint8_t*, uint8_t*> GeMM(uint8_t * Ap, uint8_t * Am, uint8_t * B, uint32_t m, uint32_t n, uint32_t k);
