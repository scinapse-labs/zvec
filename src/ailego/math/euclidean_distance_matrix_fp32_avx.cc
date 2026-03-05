// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ailego/internal/cpu_features.h>
#include "distance_matrix_accum_fp32.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE SSD_FP32_SSE
#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX

//! Calculate sum of squared difference (GENERAL)
#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate sum of squared difference (SSE)
#define SSD_FP32_SSE(xmm_m, xmm_q, xmm_sum)        \
  {                                                \
    __m128 xmm_d = _mm_sub_ps(xmm_m, xmm_q);       \
    xmm_sum = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum); \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_FP32_AVX(ymm_m, ymm_q, ymm_sum)           \
  {                                                   \
    __m256 ymm_d = _mm256_sub_ps(ymm_m, ymm_q);       \
    ymm_sum = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum); \
  }

#if defined(__AVX__)
float SquaredEuclideanDistanceAVX(const float *lhs, const float *rhs,
                                  size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 0), _mm256_load_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 0), _mm256_loadu_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 8), _mm256_loadu_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      SSD_FP32_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      SSD_FP32_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      SSD_FP32_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      SSD_FP32_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      SSD_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

#endif  // __AVX__

}  // namespace ailego
}  // namespace zvec