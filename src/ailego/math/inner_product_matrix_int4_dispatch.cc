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
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX2__)
float InnerProductAVX2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductAVX2_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size);

float MinusInnerProductAVX2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductAVX2_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size);
#endif

#if defined(__SSE4_1__)
float InnerProductSSE(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float InnerProductSSE_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size);

float MinusInnerProductSSE(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_2X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_2X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_4X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_4X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_4X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_8X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_8X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_8X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_8X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_16X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_16X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_16X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_16X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_16X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_32X1(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_32X2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_32X4(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_32X8(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_32X16(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductSSE_32X32(const uint8_t *lhs, const uint8_t *rhs, size_t size);
#endif

#if defined(__SSE4_1__)
//! Compute the distance between matrix and query (INT4, M=1, N=1)
void InnerProductMatrix<uint8_t, 1, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (dim > 63) {
    *out = InnerProductAVX2(m, q, dim >> 1);
    return;
  }
#endif  // __AVX2__
  *out = InnerProductSSE(m, q, dim >> 1);
}

//! Compute the distance between matrix and query (INT4, M=2, N=1)
void InnerProductMatrix<uint8_t, 2, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_2X1(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_2X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=2, N=2)
void InnerProductMatrix<uint8_t, 2, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_2X2(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_2X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=1)
void InnerProductMatrix<uint8_t, 4, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_4X1(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_4X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=2)
void InnerProductMatrix<uint8_t, 4, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_4X2(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_4X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=4)
void InnerProductMatrix<uint8_t, 4, 4>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_4X4(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_4X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=1)
void InnerProductMatrix<uint8_t, 8, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_8X1(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_8X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=2)
void InnerProductMatrix<uint8_t, 8, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_8X2(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_8X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=4)
void InnerProductMatrix<uint8_t, 8, 4>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_8X4(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_8X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=8)
void InnerProductMatrix<uint8_t, 8, 8>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_8X8(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_8X8(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=1)
void InnerProductMatrix<uint8_t, 16, 1>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_16X1(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_16X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=2)
void InnerProductMatrix<uint8_t, 16, 2>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_16X2(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_16X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=4)
void InnerProductMatrix<uint8_t, 16, 4>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_16X4(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_16X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=8)
void InnerProductMatrix<uint8_t, 16, 8>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_16X8(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_16X8(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=16)
void InnerProductMatrix<uint8_t, 16, 16>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_16X16(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_16X16(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=1)
void InnerProductMatrix<uint8_t, 32, 1>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_32X1(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_32X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=2)
void InnerProductMatrix<uint8_t, 32, 2>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_32X2(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_32X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=4)
void InnerProductMatrix<uint8_t, 32, 4>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_32X4(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_32X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=8)
void InnerProductMatrix<uint8_t, 32, 8>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_32X8(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_32X8(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=16)
void InnerProductMatrix<uint8_t, 32, 16>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_32X16(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_32X16(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=32)
void InnerProductMatrix<uint8_t, 32, 32>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductAVX2_32X32(m, q, dim);
    return;
  }
#else
  *out = InnerProductSSE_32X32(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=1, N=1)
void MinusInnerProductMatrix<uint8_t, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (dim > 31) {
    *out = MinusInnerProductAVX2(m, q, dim);
    return;
  }
#endif  // __AVX2__
  *out = MinusInnerProductSSE(m, q, dim);
}

//! Compute the distance between matrix and query (INT4, M=2, N=1)
void MinusInnerProductMatrix<uint8_t, 2, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_2X1(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_2X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=2, N=2)
void MinusInnerProductMatrix<uint8_t, 2, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_2X2(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_2X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=1)
void MinusInnerProductMatrix<uint8_t, 4, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_4X1(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_4X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=2)
void MinusInnerProductMatrix<uint8_t, 4, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_4X2(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_4X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=4, N=4)
void MinusInnerProductMatrix<uint8_t, 4, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_4X4(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_4X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=1)
void MinusInnerProductMatrix<uint8_t, 8, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_8X1(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_8X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=2)
void MinusInnerProductMatrix<uint8_t, 8, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_8X2(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_8X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=4)
void MinusInnerProductMatrix<uint8_t, 8, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_8X4(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_8X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=8, N=8)
void MinusInnerProductMatrix<uint8_t, 8, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_8X8(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_8X8(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=1)
void MinusInnerProductMatrix<uint8_t, 16, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_16X1(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_16X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=2)
void MinusInnerProductMatrix<uint8_t, 16, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_16X2(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_16X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=4)
void MinusInnerProductMatrix<uint8_t, 16, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_16X4(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_16X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=8)
void MinusInnerProductMatrix<uint8_t, 16, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_16X8(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_16X8(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=16, N=16)
void MinusInnerProductMatrix<uint8_t, 16, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_16X16(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_16X16(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=1)
void MinusInnerProductMatrix<uint8_t, 32, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_32X1(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_32X1(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=2)
void MinusInnerProductMatrix<uint8_t, 32, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_32X2(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_32X2(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=4)
void MinusInnerProductMatrix<uint8_t, 32, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_32X4(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_32X4(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=8)
void MinusInnerProductMatrix<uint8_t, 32, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_32X8(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_32X8(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=16)
void MinusInnerProductMatrix<uint8_t, 32, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_32X16(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_32X16(m, q, dim);
#endif  // __AVX2__
}

//! Compute the distance between matrix and query (INT4, M=32, N=32)
void MinusInnerProductMatrix<uint8_t, 32, 32>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductAVX2_32X32(m, q, dim);
    return;
  }
#else
  *out = MinusInnerProductSSE_32X32(m, q, dim);
#endif  // __AVX2__
}
#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec