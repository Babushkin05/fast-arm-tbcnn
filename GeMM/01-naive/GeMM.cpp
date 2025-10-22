#include <cstdint>

// A - m x n, B - n x k; result - m x k
int32_t *GeMM(int32_t *A, int32_t *B, u_int32_t m, u_int32_t n,
               u_int32_t k) {

  int32_t *result = new int32_t[m * k];

  for (u_int32_t i = 0; i < m; i++) {
    for (u_int32_t j = 0; j < k; j++) {
      int32_t sum = 0;
      for (u_int32_t l = 0; l < n; l++) {
        sum += A[i * n + l] * B[l * k + j];
      }
      result[i * k + j] = sum;
    }
  }
  return result;
}