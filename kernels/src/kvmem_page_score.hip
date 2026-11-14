// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// KVMem layerwise current-layer page-softmax scorer (device half).
//
// Faithful port of the qualified KVMem core `PageSoftmaxCuda` kernels
// (kvmem/src/cuda/page_softmax.cu) used by `score_current_layer` with Mean
// GQA aggregation. The core library is left ENTIRELY untouched (KVMEM_ROCM
// stays OFF); hipfire owns this HIP module and drives it directly from Rust.
//
// Four kernels, one per stage of the per-(row,head) page softmax:
//   1. kvmem_pack_current_query : native Q (F32/F16/BF16, byte strides) -> half,
//      flagging any nonfinite / FP16-overflowing operand into `invalid`.
//   2. kvmem_page_logits        : 4 pages/block, 32 lanes/page warp dot of the
//      packed Q against the canonical Mean-K, scaled -> logits[row,page].
//   3. kvmem_normalize_pages    : 128-thread shared-mem max + exp-sum per
//      (row,head) distribution -> norms[2*row]={max, 1/sum}.
//   4. kvmem_accumulate_pages   : sum_row softmax(logits) * (1/query_heads)
//      -> output[page] (the per-page attention mass).
//
// Operands cross the explicit FP16 boundary exactly as the core does: Mean-K is
// rounded to half on the host before upload, Q is rounded to half by the pack
// kernel, so every dot product is exact in float32. The only deviation from the
// core kernel bodies is the wavefront reduction: `__shfl_down(..., 32)` pins an
// explicit 32-lane segment so the 32-lanes-per-page layout is correct on both
// wave32 (gfx1100 native) and wave64 parts, instead of CUDA's warpSize==32
// assumption. The reduction order (16,8,4,2,1) is unchanged, so the result is
// bit-identical to the core on wave32.
//
// Grid/block contracts (must match crates/rdna-compute/src/kvmem_page_score.rs):
//   pack      : grid [ceil(rows*heads*dim/256)], block [256]
//   logits    : grid [ceil(blocks/4), rows*heads], block [128]
//   normalize : grid [rows*heads], block [128]
//   accumulate: grid [ceil(blocks/128)], block [128]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

// scalar ordinals mirror kvmem::NativeQueryScalar / hk_native_query_scalar.
#define KVMEM_QUERY_F32 0
#define KVMEM_QUERY_F16 1
#define KVMEM_QUERY_BF16 2

// 32-lane tree reduction, explicit width so a logical page (32 consecutive
// threads) reduces correctly whether the hardware wavefront is 32 or 64 wide.
__device__ __forceinline__ float kvmem_page_warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down(value, offset, 32);
    return value;
}

extern "C" __launch_bounds__(128)
__global__ void kvmem_page_logits(const __half* __restrict__ queries,
                                  const __half* __restrict__ means,
                                  float* __restrict__ logits,
                                  int blocks, int qheads, int kvheads, int dim,
                                  float scale) {
    const int page = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    const int lane = int(threadIdx.x) % 32;
    const int distribution = int(blockIdx.y);
    const int head = distribution % qheads;
    const int kvhead = head / (qheads / kvheads);
    if (page >= blocks) return;
    const __half* q = queries + size_t(distribution) * dim;
    const __half* k = means + (size_t(page) * kvheads + kvhead) * dim;
    float dot = 0;
    for (int d = lane; d < dim; d += 32)
        dot = fmaf(__half2float(q[d]), __half2float(k[d]), dot);
    dot = kvmem_page_warp_sum(dot);
    if (lane == 0) logits[size_t(distribution) * blocks + page] = dot * scale;
}

extern "C" __launch_bounds__(128)
__global__ void kvmem_normalize_pages(const float* __restrict__ logits,
                                      float* __restrict__ norms, int blocks) {
    __shared__ float reduction[128];
    const int tid = int(threadIdx.x), row = int(blockIdx.x);
    float maximum = -INFINITY;
    for (int b = tid; b < blocks; b += 128)
        maximum = fmaxf(maximum, logits[size_t(row) * blocks + b]);
    reduction[tid] = maximum;
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0;
    for (int b = tid; b < blocks; b += 128)
        sum += expf(logits[size_t(row) * blocks + b] - maximum);
    reduction[tid] = sum;
    __syncthreads();
    for (int s = 64; s; s /= 2) {
        if (tid < s) reduction[tid] += reduction[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        norms[2 * row] = maximum;
        norms[2 * row + 1] = 1 / reduction[0];
    }
}

extern "C" __launch_bounds__(128)
__global__ void kvmem_accumulate_pages(const float* __restrict__ logits,
                                       const float* __restrict__ norms,
                                       float* __restrict__ output,
                                       int blocks, int distributions, float weight) {
    const int b = int(blockIdx.x) * 128 + int(threadIdx.x);
    if (b >= blocks) return;
    float sum = 0;
    for (int row = 0; row < distributions; ++row)
        sum += expf(logits[size_t(row) * blocks + b] - norms[2 * row]) * norms[2 * row + 1];
    output[b] += sum * weight;
}

// Only operand transport / FP16 materialization is new relative to the legacy
// scorer; the logits/normalize/accumulate stages above reuse the qualified math.
// `invalid` borrows the first word of the norms workspace (memset to 0 before
// the launch, read back before the normalizer overwrites it).
extern "C" __launch_bounds__(256)
__global__ void kvmem_pack_current_query(const unsigned char* __restrict__ input,
                                         __half* __restrict__ output,
                                         unsigned* __restrict__ invalid,
                                         unsigned rows, unsigned heads, unsigned dim,
                                         size_t row_stride, size_t head_stride,
                                         size_t dimension_stride, int scalar,
                                         int finite_half_required) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = size_t(rows) * heads * dim;
    if (index >= count) return;
    const size_t d = index % dim, head = (index / dim) % heads,
                 row = index / (size_t(dim) * heads);
    const unsigned char* source =
        input + row * row_stride + head * head_stride + d * dimension_stride;
    float value;
    if (scalar == KVMEM_QUERY_F32)
        value = *reinterpret_cast<const float*>(source);
    else if (scalar == KVMEM_QUERY_F16)
        value = __half2float(*reinterpret_cast<const __half*>(source));
    else
        value = __uint_as_float(unsigned(*reinterpret_cast<const unsigned short*>(source)) << 16);
    const __half represented = __float2half_rn(value);
    if (!isfinite(value) || (finite_half_required && !isfinite(__half2float(represented))))
        atomicExch(invalid, 1u);
    output[index] = represented;
}