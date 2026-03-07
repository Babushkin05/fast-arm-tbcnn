#include <cstdint>

// A - m x n, B - n x k; result - m x k
std::int32_t *GeMM(std::int32_t *A, std::int32_t *B, std::uint32_t m, std::uint32_t n,
               std::uint32_t k) {

  std::int32_t *result = new std::int32_t[m * k];

  for (std::uint32_t i = 0; i < m; i++) {
    for (std::uint32_t j = 0; j < k; j++) {
      std::int32_t sum = 0;
      for (std::uint32_t l = 0; l < n; l++) {
        sum += A[i * n + l] * B[l * k + j];
      }
      result[i * k + j] = sum;
    }
  }
  return result;
}
