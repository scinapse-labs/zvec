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

#include "distance_matrix_accum_int8.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_INT8_STEP_SSE SSD_INT8_SSE
#define ACCUM_INT8_STEP_AVX SSD_INT8_AVX

#if defined(__SSE4_1__)
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
#endif  // __SSE4_1__

#if defined(__AVX2__)
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
#endif  // __AVX2__

//! Calculate sum of squared difference (GENERAL)
#define SSD_INT8_GENERAL(m, q, sum)   \
  {                                   \
    int32_t x = m - q;                \
    sum += static_cast<float>(x * x); \
  }

//! Calculate sum of squared difference (SSE)
#define SSD_INT8_SSE(xmm_m, xmm_q, xmm_sum)                                \
  {                                                                        \
    xmm_sum = _mm_add_epi32(                                               \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_m),              \
                                         _mm_sign_epi8(xmm_m, xmm_m)),     \
                       ONES_INT16_SSE),                                    \
        xmm_sum);                                                          \
    xmm_sum = _mm_add_epi32(                                               \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_q),              \
                                         _mm_sign_epi8(xmm_q, xmm_q)),     \
                       ONES_INT16_SSE),                                    \
        xmm_sum);                                                          \
    xmm_sum = _mm_sub_epi32(                                               \
        xmm_sum,                                                           \
        _mm_slli_epi32(                                                    \
            _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_q),          \
                                             _mm_sign_epi8(xmm_m, xmm_q)), \
                           ONES_INT16_SSE),                                \
            1));                                                           \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_INT8_AVX(ymm_m, ymm_q, ymm_sum)                                    \
  {                                                                            \
    ymm_sum = _mm256_add_epi32(                                                \
        _mm256_madd_epi16(                                                     \
            _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_m),                       \
                                 _mm256_sign_epi8(ymm_m, ymm_m)),              \
            ONES_INT16_AVX),                                                   \
        ymm_sum);                                                              \
    ymm_sum = _mm256_add_epi32(                                                \
        _mm256_madd_epi16(                                                     \
            _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_q),                       \
                                 _mm256_sign_epi8(ymm_q, ymm_q)),              \
            ONES_INT16_AVX),                                                   \
        ymm_sum);                                                              \
    ymm_sum = _mm256_sub_epi32(                                                \
        ymm_sum, _mm256_slli_epi32(                                            \
                     _mm256_madd_epi16(                                        \
                         _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_q),          \
                                              _mm256_sign_epi8(ymm_m, ymm_q)), \
                         ONES_INT16_AVX),                                      \
                     1));                                                      \
  }

//! Compute the square root of value (SSE)
#define SQRT_FP32_SSE(v, ...) _mm_sqrt_ps(_mm_cvtepi32_ps(v))

//! Compute the square root of value (AVX)
#define SQRT_FP32_AVX(v, ...) _mm256_sqrt_ps(_mm256_cvtepi32_ps(v))

#if defined(__AVX2__)
//! Squared Euclidean Distance
float SquaredEuclideanDistanceAVX22(const int8_t *lhs,
                                                const int8_t *rhs,
                                                size_t size) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);
  float result = 0.0;

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));

      __m256i ymm_d = _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_0, ymm_rhs_0),
                                      _mm256_min_epi8(ymm_lhs_0, ymm_rhs_0));
      ymm_lhs_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ymm_d));
      ymm_rhs_0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ymm_d, 1));
      ymm_sum_0 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_lhs_0, ymm_lhs_0), ymm_sum_0);
      ymm_sum_1 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_rhs_0, ymm_rhs_0), ymm_sum_1);

      ymm_d = _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_1, ymm_rhs_1),
                              _mm256_min_epi8(ymm_lhs_1, ymm_rhs_1));
      ymm_lhs_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ymm_d));
      ymm_rhs_1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ymm_d, 1));
      ymm_sum_0 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_lhs_1, ymm_lhs_1), ymm_sum_0);
      ymm_sum_1 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_rhs_1, ymm_rhs_1), ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      __m256i ymm_d = _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs, ymm_rhs),
                                      _mm256_min_epi8(ymm_lhs, ymm_rhs));
      ymm_lhs = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ymm_d));
      ymm_rhs = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ymm_d, 1));
      ymm_sum_0 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_lhs, ymm_lhs), ymm_sum_0);
      ymm_sum_1 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_rhs, ymm_rhs), ymm_sum_1);
      lhs += 32;
      rhs += 32;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));

      __m256i ymm_d = _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_0, ymm_rhs_0),
                                      _mm256_min_epi8(ymm_lhs_0, ymm_rhs_0));
      ymm_lhs_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ymm_d));
      ymm_rhs_0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ymm_d, 1));
      ymm_sum_0 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_lhs_0, ymm_lhs_0), ymm_sum_0);
      ymm_sum_1 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_rhs_0, ymm_rhs_0), ymm_sum_1);

      ymm_d = _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_1, ymm_rhs_1),
                              _mm256_min_epi8(ymm_lhs_1, ymm_rhs_1));
      ymm_lhs_1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ymm_d));
      ymm_rhs_1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ymm_d, 1));
      ymm_sum_0 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_lhs_1, ymm_lhs_1), ymm_sum_0);
      ymm_sum_1 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_rhs_1, ymm_rhs_1), ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      __m256i ymm_d = _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs, ymm_rhs),
                                      _mm256_min_epi8(ymm_lhs, ymm_rhs));
      ymm_lhs = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(ymm_d));
      ymm_rhs = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ymm_d, 1));
      ymm_sum_0 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_lhs, ymm_lhs), ymm_sum_0);
      ymm_sum_1 =
          _mm256_add_epi32(_mm256_madd_epi16(ymm_rhs, ymm_rhs), ymm_sum_1);
      lhs += 32;
      rhs += 32;
    }
  }
  result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));

  if (last >= lhs + 16) {
    __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
    __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
    __m128i xmm_sum = _mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),
                                   _mm_min_epi8(xmm_lhs, xmm_rhs));
    xmm_lhs = _mm_cvtepu8_epi16(xmm_sum);
    xmm_rhs = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_sum, xmm_sum));
    xmm_sum = _mm_add_epi32(_mm_madd_epi16(xmm_lhs, xmm_lhs),
                            _mm_madd_epi16(xmm_rhs, xmm_rhs));
    result += static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum));
    lhs += 16;
    rhs += 16;
  }
  switch (last - lhs) {
    case 15:
      SSD_INT8_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      SSD_INT8_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      SSD_INT8_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      SSD_INT8_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      SSD_INT8_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      SSD_INT8_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      SSD_INT8_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      SSD_INT8_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      SSD_INT8_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      SSD_INT8_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      SSD_INT8_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      SSD_INT8_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      SSD_INT8_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_INT8_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_INT8_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

#endif  // __AVX2__

}  // namespace ailego
}  // namespace zvec