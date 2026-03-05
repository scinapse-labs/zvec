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

#if defined(__SSE4_1__)
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
#endif  // __SSE4_1__

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

//! Compute the square root of value (SSE)
#define SQRT_FP32_SSE(v, ...) _mm_sqrt_ps(_mm_cvtepi32_ps(v))

#if defined(__SSE4_1__)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceSSE(const int8_t *lhs,
                                                const int8_t *rhs,
                                                size_t size) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 5) << 5);

  __m128i xmm_sum_0 = _mm_setzero_si128();
  __m128i xmm_sum_1 = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m128i xmm_lhs_0 = _mm_load_si128((const __m128i *)(lhs + 0));
      __m128i xmm_lhs_1 = _mm_load_si128((const __m128i *)(lhs + 16));
      __m128i xmm_rhs_0 = _mm_load_si128((const __m128i *)(rhs + 0));
      __m128i xmm_rhs_1 = _mm_load_si128((const __m128i *)(rhs + 16));

      __m128i xmm_d = _mm_sub_epi8(_mm_max_epi8(xmm_lhs_0, xmm_rhs_0),
                                   _mm_min_epi8(xmm_lhs_0, xmm_rhs_0));
      xmm_lhs_0 = _mm_cvtepu8_epi16(xmm_d);
      xmm_rhs_0 = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_d, xmm_d));
      xmm_d = _mm_sub_epi8(_mm_max_epi8(xmm_lhs_1, xmm_rhs_1),
                           _mm_min_epi8(xmm_lhs_1, xmm_rhs_1));
      xmm_lhs_1 = _mm_cvtepu8_epi16(xmm_d);
      xmm_rhs_1 = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_d, xmm_d));

      xmm_sum_0 =
          _mm_add_epi32(_mm_madd_epi16(xmm_lhs_0, xmm_lhs_0), xmm_sum_0);
      xmm_sum_1 =
          _mm_add_epi32(_mm_madd_epi16(xmm_rhs_0, xmm_rhs_0), xmm_sum_1);
      xmm_sum_0 =
          _mm_add_epi32(_mm_madd_epi16(xmm_lhs_1, xmm_lhs_1), xmm_sum_0);
      xmm_sum_1 =
          _mm_add_epi32(_mm_madd_epi16(xmm_rhs_1, xmm_rhs_1), xmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      __m128i xmm_d = _mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),
                                   _mm_min_epi8(xmm_lhs, xmm_rhs));
      xmm_lhs = _mm_cvtepu8_epi16(xmm_d);
      xmm_rhs = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_d, xmm_d));
      xmm_sum_0 = _mm_add_epi32(_mm_madd_epi16(xmm_lhs, xmm_lhs), xmm_sum_0);
      xmm_sum_1 = _mm_add_epi32(_mm_madd_epi16(xmm_rhs, xmm_rhs), xmm_sum_1);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m128i xmm_lhs_0 = _mm_loadu_si128((const __m128i *)(lhs + 0));
      __m128i xmm_lhs_1 = _mm_loadu_si128((const __m128i *)(lhs + 16));
      __m128i xmm_rhs_0 = _mm_loadu_si128((const __m128i *)(rhs + 0));
      __m128i xmm_rhs_1 = _mm_loadu_si128((const __m128i *)(rhs + 16));

      __m128i xmm_d = _mm_sub_epi8(_mm_max_epi8(xmm_lhs_0, xmm_rhs_0),
                                   _mm_min_epi8(xmm_lhs_0, xmm_rhs_0));
      xmm_lhs_0 = _mm_cvtepu8_epi16(xmm_d);
      xmm_rhs_0 = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_d, xmm_d));
      xmm_d = _mm_sub_epi8(_mm_max_epi8(xmm_lhs_1, xmm_rhs_1),
                           _mm_min_epi8(xmm_lhs_1, xmm_rhs_1));
      xmm_lhs_1 = _mm_cvtepu8_epi16(xmm_d);
      xmm_rhs_1 = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_d, xmm_d));

      xmm_sum_0 =
          _mm_add_epi32(_mm_madd_epi16(xmm_lhs_0, xmm_lhs_0), xmm_sum_0);
      xmm_sum_1 =
          _mm_add_epi32(_mm_madd_epi16(xmm_rhs_0, xmm_rhs_0), xmm_sum_1);
      xmm_sum_0 =
          _mm_add_epi32(_mm_madd_epi16(xmm_lhs_1, xmm_lhs_1), xmm_sum_0);
      xmm_sum_1 =
          _mm_add_epi32(_mm_madd_epi16(xmm_rhs_1, xmm_rhs_1), xmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      __m128i xmm_d = _mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),
                                   _mm_min_epi8(xmm_lhs, xmm_rhs));
      xmm_lhs = _mm_cvtepu8_epi16(xmm_d);
      xmm_rhs = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(xmm_d, xmm_d));
      xmm_sum_0 = _mm_add_epi32(_mm_madd_epi16(xmm_lhs, xmm_lhs), xmm_sum_0);
      xmm_sum_1 = _mm_add_epi32(_mm_madd_epi16(xmm_rhs, xmm_rhs), xmm_sum_1);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(
      HorizontalAdd_INT32_V128(_mm_add_epi32(xmm_sum_0, xmm_sum_1)));

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


//! SquaredEuclideanDistance
float SquaredEuclideanDistanceSSE_2X1(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_2X1_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_2X2(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_2X2_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}
float SquaredEuclideanDistanceSSE_4X1(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_4X1_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}
float SquaredEuclideanDistanceSSE_4X2(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_4X2_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_4X4(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_4X4_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_8X1(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_8X1_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_8X2(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_8X2_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_8X4(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_8X4_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_8X8(const int8_t *lhs, const int8_t *rhs,
                                      size_t size) {
  float score;
  ACCUM_INT8_8X8_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_16X1(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_16X1_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_16X2(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_16X2_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_16X4(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_16X4_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_16X8(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_16X8_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_16X16(const int8_t *lhs, const int8_t *rhs,
                                        size_t size) {
  float score;
  ACCUM_INT8_16X16_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_32X1(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_32X1_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_32X2(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_32X2_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_32X4(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_32X4_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_32X8(const int8_t *lhs, const int8_t *rhs,
                                       size_t size) {
  float score;
  ACCUM_INT8_32X8_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_32X16(const int8_t *lhs, const int8_t *rhs,
                                        size_t size) {
  float score;
  ACCUM_INT8_32X16_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

float SquaredEuclideanDistanceSSE_32X32(const int8_t *lhs, const int8_t *rhs,
                                        size_t size) {
  float score;
  ACCUM_INT8_32X32_SSE(lhs, rhs, size, &score, _mm_cvtepi32_ps)

  return score;
}

//! EuclideanDistance
float EuclideanDistanceSSE_2X1(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_2X1_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_2X2(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_2X2_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}
float EuclideanDistanceSSE_4X1(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_4X1_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}
float EuclideanDistanceSSE_4X2(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_4X2_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_4X4(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_4X4_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_8X1(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_8X1_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_8X2(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_8X2_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_8X4(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_8X4_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_8X8(const int8_t *lhs, const int8_t *rhs,
                               size_t size) {
  float score;
  ACCUM_INT8_8X8_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_16X1(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_16X1_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_16X2(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_16X2_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_16X4(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_16X4_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_16X8(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_16X8_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_16X16(const int8_t *lhs, const int8_t *rhs,
                                 size_t size) {
  float score;
  ACCUM_INT8_16X16_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_32X1(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_32X1_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_32X2(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_32X2_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_32X4(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_32X4_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_32X8(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  float score;
  ACCUM_INT8_32X8_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_32X16(const int8_t *lhs, const int8_t *rhs,
                                 size_t size) {
  float score;
  ACCUM_INT8_32X16_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

float EuclideanDistanceSSE_32X32(const int8_t *lhs, const int8_t *rhs,
                                 size_t size) {
  float score;
  ACCUM_INT8_32X32_SSE(lhs, rhs, size, &score, SQRT_FP32_SSE)

  return score;
}

#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec