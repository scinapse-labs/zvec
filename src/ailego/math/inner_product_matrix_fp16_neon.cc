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
#include "distance_matrix_accum_fp16.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_NEON FMA_FP32_NEON
#define ACCUM_FP16_STEP_GENERAL FMA_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON FMA_FP16_NEON

//! Reverse sign of value (GENERAL)
#define NEGATE_FP32_GENERAL(v) -(v)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP16_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP16_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f16(v_sum, v_m, v_q);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP32_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_q);

#if defined(__ARM_NEON)
float InnerProductNEON(const Float16 *lhs,const Float16 *rhs, size_t size) {
  float score;

  ACCUM_FP16_1X1_NEON(lhs, rhs, size, &score, 0ull, )

  return score;
}

float MinusInnerProductNEON(const Float16 *lhs,const Float16 *rhs, size_t size) {
  float score;

  ACCUM_FP16_1X1_NEON(lhs, rhs, size, &score, 0ull, NEGATE_FP32_GENERAL)

  return score;
}
#endif

}  // namespace ailego
}  // namespace zvec