// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// KVMem layerwise pressure attention-mass tap (device half).
//
// Faithful HIP port of ninfer's `attention_page_mass` recompute kernel
// (ninfer src/ops/token_sums/attention_page_mass.cu). It samples a target
// layer's native Q against the represented persistent Q8_0 K cache and returns
// the per-page attention mass, normalized across the COMPLETE visible cache and
// averaged over query heads. It is a SEPARATE K-only recompute: no V is read, no
// attention kernel is modified, no allocation / host copy / state mutation happens
// here. The legacy attention path is therefore byte-neutral.
//
// Three kernels, one per stage (mirrors ninfer page_lse / normalize_lse /
// average_heads):
//   1. kvmem_page_mass_lse       : per (page, sample*head) block, 128 threads.
//      Each lane owns 8 query dims (lane+32*i); decodes the hipfire Q8_0 K row
//      (f16 scale + 32 int8 codes per 32-dim block, inline, NO Hadamard), dots
//      against Q, reduces the 32-lane segment, scales, and emits the page's
//      token logsumexp into scratch[sample*head, page].
//   2. kvmem_page_mass_normalize : per (sample*head) distribution, shared-mem
//      max + exp-sum across all pages -> norms[2*row]={max, 1/sum}.
//   3. kvmem_page_mass_average   : per (page, sample), sum_head softmax(lse)
//      / query_heads -> output[sample, page] (the per-page attention mass).
//
// Adaptations from ninfer (geometry only, algorithm unchanged):
//   * page_tokens is a runtime parameter (hipfire 128, ninfer 64). The shared
//     reduction is sized for up to 128 tokens/page.
//   * K coding is hipfire plain q8_0 ONLY: per token row = kv_heads * seg bytes,
//     seg = (head_dim/32)*34; block b holds an f16 scale at 34*b then 32 int8
//     codes at 34*b+2. No Hadamard rotation, no NVFP4 / rotated-int4 branches.
//   * Q is read in its native scalar (F32/F16/BF16) and converted to float; the
//     dot is exact in float32. head_dim is fixed at 256 (8 dims/lane).
//   * wavefront reduction uses an explicit 32-lane segment (`__shfl_down(v,o,32)`)
//     so the 32-lanes-per-dot layout is correct on wave32 (gfx1100 native) and
//     wave64 parts, matching the sibling kvmem_page_score.hip convention.
//
// Grid/block contracts (must match crates/rdna-compute/src/kvmem_page_mass.rs):
//   lse       : grid [pages, samples*heads], block [128]
//   normalize : grid [samples*heads],        block [128]
//   average   : grid [ceil(pages/128), samples], block [128]
//
// Operand contract: q is contiguous [rows, heads, dim] (dim innermost) in the
// native scalar; rows/positions are I32[samples] selecting Q columns and inclusive
// compact causal ends (positions in [0, pages*page_tokens)); table is the immutable
// I32[pages] block table row (physical page per logical page, -1 = unmapped);
// keys is the page-major Q8_0 K cache (physical_pages pages). scratch[norms/output]
// are disjoint caller-owned FP32 buffers. An unmapped/out-of-range VISIBLE physical
// page yields NaN, which the owner must reject before accepting statistics.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

// scalar ordinals mirror kvmem::NativeQueryScalar / hk_native_query_scalar.
#define KVMEM_QUERY_F32 0
#define KVMEM_QUERY_F16 1
#define KVMEM_QUERY_BF16 2

#define MASS_DIM 256
#define MASS_BLK 34
#define MASS_SEG (MASS_DIM / 32 * MASS_BLK)

__device__ __forceinline__ float mass_f16(unsigned short b) {
    return (float)__builtin_bit_cast(_Float16, b);
}

__device__ __forceinline__ float mass_bf16(unsigned short b) {
    const unsigned int bits = ((unsigned int)b) << 16;
    return __builtin_bit_cast(float, bits);
}

__device__ __forceinline__ float mass_query(const void* q, size_t index, int scalar) {
    if (scalar == KVMEM_QUERY_F32)
        return static_cast<const float*>(q)[index];
    if (scalar == KVMEM_QUERY_F16)
        return mass_f16(static_cast<const unsigned short*>(q)[index]);
    return mass_bf16(static_cast<const unsigned short*>(q)[index]);
}

// 32-lane tree reduction, explicit width so a logical 32-lane dot segment reduces
// correctly whether the hardware wavefront is 32 or 64 wide.
__device__ __forceinline__ float mass_warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down(value, offset, 32);
    return value;
}

extern "C" __launch_bounds__(128)
__global__ void kvmem_page_mass_lse(const void* __restrict__ q,
                                    const int* __restrict__ rows,
                                    const int* __restrict__ positions,
                                    const int* __restrict__ table,
                                    const void* __restrict__ keys,
                                    int qheads, int kvheads, int pages,
                                    int physical_pages, int page_tokens, int dim,
                                    int scalar, float scale,
                                    float* __restrict__ output) {
    __shared__ float values[128];
    const int page = int(blockIdx.x), distribution = int(blockIdx.y);
    const int sample = distribution / qheads, head = distribution % qheads;
    const int tid = int(threadIdx.x), lane = tid & 31, warp = tid / 32;
    const int visible = positions[sample] + 1;
    if (page * page_tokens >= visible) {
        if (tid == 0) output[size_t(distribution) * pages + page] = -INFINITY;
        return;
    }
    float query[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        const size_t qi = (size_t(rows[sample]) * qheads + head) * dim + d;
        query[i] = mass_query(q, qi, scalar);
    }
    const int physical = table[page], kvhead = head / (qheads / kvheads);
    if (physical < 0 || physical >= physical_pages) {
        if (tid == 0) output[size_t(distribution) * pages + page] = NAN;
        return;
    }
    const int row_stride = kvheads * MASS_SEG;
    for (int offset = warp; offset < page_tokens; offset += 4) {
        float dot = 0;
        const bool live = page * page_tokens + offset < visible;
        if (live) {
            const unsigned char* kbase = static_cast<const unsigned char*>(keys) +
                (size_t(physical) * page_tokens + offset) * row_stride + size_t(kvhead) * MASS_SEG;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                // dim d = lane + 32*i lives in block i at code position lane.
                const float ks = mass_f16(*reinterpret_cast<const unsigned short*>(kbase + MASS_BLK * i));
                const float code = (float)(int)(signed char)kbase[MASS_BLK * i + 2 + lane];
                dot = __builtin_fmaf(query[i], code * ks, dot);
            }
            dot = mass_warp_sum(dot);
        }
        if (lane == 0) values[offset] = live ? dot * scale : -INFINITY;
    }
    __syncthreads();
    float value = tid < page_tokens ? values[tid] : -INFINITY;
    values[tid] = value; __syncthreads();
    for (int s = 64; s; s /= 2) { if (tid < s) values[tid] = fmaxf(values[tid], values[tid + s]); __syncthreads(); }
    const float maximum = values[0];
    __syncthreads(); // Every warp must read the maximum before values[0] becomes a sum.
    value = isfinite(value) ? expf(value - maximum) : 0;
    values[tid] = value; __syncthreads();
    for (int s = 64; s; s /= 2) { if (tid < s) values[tid] += values[tid + s]; __syncthreads(); }
    if (tid == 0)
        output[size_t(distribution) * pages + page] = values[0] > 0 ? maximum + logf(values[0]) : -INFINITY;
}

extern "C" __launch_bounds__(128)
__global__ void kvmem_page_mass_normalize(const float* __restrict__ lse, int pages,
                                          float* __restrict__ norms) {
    __shared__ float sums[128];
    const int tid = int(threadIdx.x), row = int(blockIdx.x);
    float maximum = -INFINITY;
    for (int p = tid; p < pages; p += 128) maximum = fmaxf(maximum, lse[size_t(row) * pages + p]);
    sums[tid] = maximum; __syncthreads();
    for (int s = 64; s; s /= 2) { if (tid < s) sums[tid] = fmaxf(sums[tid], sums[tid + s]); __syncthreads(); }
    maximum = sums[0];
    __syncthreads();
    float total = 0;
    for (int p = tid; p < pages; p += 128) total += expf(lse[size_t(row) * pages + p] - maximum);
    sums[tid] = total; __syncthreads();
    for (int s = 64; s; s /= 2) { if (tid < s) sums[tid] += sums[tid + s]; __syncthreads(); }
    if (tid == 0) { norms[2 * row] = maximum; norms[2 * row + 1] = 1 / sums[0]; }
}

extern "C" __launch_bounds__(128)
__global__ void kvmem_page_mass_average(const float* __restrict__ lse,
                                        const float* __restrict__ norms, int pages,
                                        int heads, float* __restrict__ output) {
    const int page = int(blockIdx.x) * 128 + int(threadIdx.x), sample = int(blockIdx.y);
    if (page >= pages) return;
    float mass = 0;
    for (int h = 0; h < heads; ++h) {
        const int row = sample * heads + h;
        mass += expf(lse[size_t(row) * pages + page] - norms[2 * row]) * norms[2 * row + 1];
    }
    output[size_t(sample) * pages + page] = mass / heads;
}