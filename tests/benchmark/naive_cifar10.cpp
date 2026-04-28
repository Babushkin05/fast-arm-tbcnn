/**
 * Naive inference benchmark — pure scalar, no optimizations.
 *
 * Used as a baseline to measure the speedup of all optimizations.
 * - Conv2D: direct 7-nested loops (N,C,H,W,KH,KW,M), no im2col
 * - GeMM: scalar triple loop, float32
 * - No NEON, no tiling, no caching, no quantization
 */
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace std;

// ============================================================================
// Naive Conv2D: direct 7-nested loops (N, M, OH, OW, C, KH, KW)
// Output: [N, M, OH, OW] = input [N, C, H, W] * weight [M, C, KH, KW] + bias [M]
// ============================================================================
vector<float> naive_conv2d(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ weight,  // [M, C, KH, KW]
    const float* __restrict__ bias,    // [M]
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t M, int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w)
{
    int64_t OH = (H + 2*pad_h - KH) / stride_h + 1;
    int64_t OW = (W + 2*pad_w - KW) / stride_w + 1;

    vector<float> output(N * M * OH * OW, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                int64_t ih = oh * stride_h - pad_h + kh;
                                int64_t iw = ow * stride_w - pad_w + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float in_val = input[((n * C + c) * H + ih) * W + iw];
                                    float w_val = weight[((m * C + c) * KH + kh) * KW + kw];
                                    sum += in_val * w_val;
                                }
                            }
                        }
                    }
                    sum += (bias != nullptr) ? bias[m] : 0.0f;
                    // ReLU
                    if (sum < 0) sum = 0.0f;
                    output[((n * M + m) * OH + oh) * OW + ow] = sum;
                }
            }
        }
    }
    return output;
}

// ============================================================================
// Naive GeMM: C[M,N] = A[M,K] * B[K,N]  (scalar float, transB=false)
// ============================================================================
vector<float> naive_gemm(
    const float* __restrict__ A, const float* __restrict__ B,
    int64_t M, int64_t K, int64_t N,
    const float* __restrict__ bias = nullptr) // bias [N]
{
    vector<float> C(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            if (bias) sum += bias[j];
            if (sum < 0) sum = 0.0f; // ReLU
            C[i * N + j] = sum;
        }
    }
    return C;
}

// ============================================================================
// Naive MaxPool2D: [N, C, H, W] -> [N, C, OH, OW]
// ============================================================================
vector<float> naive_maxpool2d(
    const float* __restrict__ input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t KH, int64_t KW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w)
{
    int64_t OH = (H + 2*pad_h - KH) / stride_h + 1;
    int64_t OW = (W + 2*pad_w - KW) / stride_w + 1;
    vector<float> output(N * C * OH * OW, -1e38f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float max_val = -1e38f;
                    for (int64_t kh = 0; kh < KH; ++kh) {
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            int64_t ih = oh * stride_h - pad_h + kh;
                            int64_t iw = ow * stride_w - pad_w + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float val = input[((n * C + c) * H + ih) * W + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    output[((n * C + c) * OH + oh) * OW + ow] = max_val;
                }
            }
        }
    }
    return output;
}

// ============================================================================
// CIFAR-10 model architecture (matching tests/train/cifar10/cifar10_tbn.onnx)
//
// Conv1: input [1,3,32,32],  weight [32,3,3,3]   -> [1,32,32,32]
// MaxPool1: [1,32,32,32]     -> [1,32,16,16]
// Conv2: input [1,32,16,16], weight [64,32,3,3]   -> [1,64,16,16]
// MaxPool2: [1,64,16,16]     -> [1,64,8,8]
// Conv3: input [1,64,8,8],   weight [128,64,3,3]  -> [1,128,8,8]
// MaxPool3: [1,128,8,8]      -> [1,128,4,4]
// Flatten: [1,128,4,4]       -> [1,2048]
// FC1: [1,2048] x [2048,256] -> [1,256]
// FC2: [1,256] x [256,10]    -> [1,10]
// ============================================================================

struct Cifar10ModelNaive {
    // Conv1
    vector<float> w_conv1;  // [32, 3, 3, 3]
    vector<float> b_conv1;  // [32]
    // Conv2
    vector<float> w_conv2;  // [64, 32, 3, 3]
    vector<float> b_conv2;  // [64]
    // Conv3
    vector<float> w_conv3;  // [128, 64, 3, 3]
    vector<float> b_conv3;  // [128]
    // FC1
    vector<float> w_fc1;    // [2048, 256]
    vector<float> b_fc1;    // [256]
    // FC2
    vector<float> w_fc2;    // [256, 10]
    vector<float> b_fc2;    // [10]

    Cifar10ModelNaive() {
        auto random_init = [](vector<float>& v) {
            for (auto& x : v) x = (float)(rand() % 1000) / 500.0f - 1.0f;
        };

        w_conv1.resize(32 * 3 * 3 * 3);    b_conv1.resize(32);
        w_conv2.resize(64 * 32 * 3 * 3);   b_conv2.resize(64);
        w_conv3.resize(128 * 64 * 3 * 3);  b_conv3.resize(128);
        w_fc1.resize(2048 * 256);          b_fc1.resize(256);
        w_fc2.resize(256 * 10);             b_fc2.resize(10);

        random_init(w_conv1); random_init(b_conv1);
        random_init(w_conv2); random_init(b_conv2);
        random_init(w_conv3); random_init(b_conv3);
        random_init(w_fc1); random_init(b_fc1);
        random_init(w_fc2); random_init(b_fc2);
    }
};

vector<float> naive_inference(const vector<float>& input_image, const Cifar10ModelNaive& model) {
    // Conv1: [1,3,32,32] -> [1,32,32,32]
    auto c1 = naive_conv2d(input_image.data(), model.w_conv1.data(), model.b_conv1.data(),
                           1, 3, 32, 32, 32, 3, 3, 1, 1, 1, 1);

    // MaxPool1: [1,32,32,32] -> [1,32,16,16]
    auto p1 = naive_maxpool2d(c1.data(), 1, 32, 32, 32, 2, 2, 2, 2, 0, 0);

    // Conv2: [1,32,16,16] -> [1,64,16,16]
    auto c2 = naive_conv2d(p1.data(), model.w_conv2.data(), model.b_conv2.data(),
                           1, 32, 16, 16, 64, 3, 3, 1, 1, 1, 1);

    // MaxPool2: [1,64,16,16] -> [1,64,8,8]
    auto p2 = naive_maxpool2d(c2.data(), 1, 64, 16, 16, 2, 2, 2, 2, 0, 0);

    // Conv3: [1,64,8,8] -> [1,128,8,8]
    auto c3 = naive_conv2d(p2.data(), model.w_conv3.data(), model.b_conv3.data(),
                           1, 64, 8, 8, 128, 3, 3, 1, 1, 1, 1);

    // MaxPool3: [1,128,8,8] -> [1,128,4,4]
    auto p3 = naive_maxpool2d(c3.data(), 1, 128, 8, 8, 2, 2, 2, 2, 0, 0);

    // Flatten: [1,128,4,4] -> [1,2048]

    // FC1: [1,2048] x [2048,256] -> [1,256]
    auto fc1 = naive_gemm(p3.data(), model.w_fc1.data(), 1, 2048, 256, model.b_fc1.data());

    // FC2: [1,256] x [256,10] -> [1,10]
    auto fc2 = naive_gemm(fc1.data(), model.w_fc2.data(), 1, 256, 10, model.b_fc2.data());

    return fc2;
}

int main() {
    srand(42);
    const int WARMUP = 2;
    const int RUNS = 10;

    Cifar10ModelNaive model;
    vector<float> input(1 * 3 * 32 * 32);
    for (auto& x : input) x = (float)(rand() % 1000) / 500.0f - 1.0f;

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
        naive_inference(input, model);

    // Benchmark
    double total_ms = 0;
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = chrono::high_resolution_clock::now();
        auto result = naive_inference(input, model);
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();
        total_ms += ms;
        cout << "Run " << (r+1) << ": " << ms << " ms" << endl;
    }

    double avg = total_ms / RUNS;
    cout << "\nNaive inference (scalar, no im2col, no NEON, no quantization)" << endl;
    cout << "Average latency: " << avg << " ms" << endl;

    // Compare with our optimized: TBN pre-quantized = 0.579 ms
    double our_optimized = 0.579;
    cout << "Optimized TBN pre-quantized: " << our_optimized << " ms" << endl;
    cout << "Speedup: " << (avg / our_optimized) << "x" << endl;

    return 0;
}
