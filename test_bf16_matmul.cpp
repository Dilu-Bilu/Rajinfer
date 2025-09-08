#include <assert.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <omp.h>

typedef uint16_t bf16_t;

// Convert float -> bf16
inline bf16_t float_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<bf16_t>(u >> 16);
}

// Convert bf16 -> float
inline float bf16_to_float(bf16_t x) {
    uint32_t u = static_cast<uint32_t>(x) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// matmul supporting bfloat16 weights by expanding to float32 before compute.
static void bf16_matmul(float* xout, const float* x, const bf16_t* w, int n, int d) {
#if defined(__AVX2__)
    if (n % 16 == 0) {
        // AVX2 path
#pragma omp parallel for
        for (int i = 0; i < d; i++) {
            __m256 sumlo = _mm256_setzero_ps();
            __m256 sumhi = _mm256_setzero_ps();
            const bf16_t* wi = w + static_cast<size_t>(i) * n;

            for (int j = 0; j < n; j += 16) {
                __m256i w16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wi + j));
                __m128i w16_lo = _mm256_extractf128_si256(w16, 0);
                __m128i w16_hi = _mm256_extractf128_si256(w16, 1);

                __m256i w32_lo = _mm256_cvtepu16_epi32(w16_lo);
                __m256i w32_hi = _mm256_cvtepu16_epi32(w16_hi);
                w32_lo = _mm256_slli_epi32(w32_lo, 16);
                w32_hi = _mm256_slli_epi32(w32_hi, 16);

                __m256 wfp_lo = _mm256_castsi256_ps(w32_lo);
                __m256 wfp_hi = _mm256_castsi256_ps(w32_hi);

                __m256 x_lo = _mm256_loadu_ps(x + j);
                __m256 x_hi = _mm256_loadu_ps(x + j + 8);

                sumlo = _mm256_fmadd_ps(wfp_lo, x_lo, sumlo);
                sumhi = _mm256_fmadd_ps(wfp_hi, x_hi, sumhi);
            }

            __m256 sum8 = _mm256_add_ps(sumlo, sumhi);
            __m128 sum4 = _mm_add_ps(_mm256_extractf128_ps(sum8, 0),
                                     _mm256_extractf128_ps(sum8, 1));
            __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xF1);
            xout[i] = _mm_cvtss_f32(sum1);
        }
        return;
    }
#endif

    // Scalar fallback
#pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        const bf16_t* wi = w + static_cast<size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            uint32_t bits = static_cast<uint32_t>(wi[j]) << 16;
            float wf;
            std::memcpy(&wf, &bits, sizeof(float));
            acc += wf * x[j];
        }
        xout[i] = acc;
    }
}

int main() {
    // Example 2x2 matmul
    float x[2] = {1.0f, 2.0f};
    float xout[2] = {0.0f, 0.0f};

    bf16_t w[4];
    w[0] = float_to_bf16(1.0f);
    w[1] = float_to_bf16(2.0f);
    w[2] = float_to_bf16(3.0f);
    w[3] = float_to_bf16(4.0f);

    bf16_matmul(xout, x, w, 2, 2);

    std::cout << "xout[0] = " << xout[0] << " (expected 5)" << std::endl;
    std::cout << "xout[1] = " << xout[1] << " (expected 11)" << std::endl;

    assert(std::abs(xout[0] - 5.0f) < 1e-3);
    assert(std::abs(xout[1] - 11.0f) < 1e-3);

    std::cout << "BF16 matmul test passed!" << std::endl;
    return 0;
}
