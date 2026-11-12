// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// KVMem statistics tap: preserve the post-RMSNorm / pre-RoPE Q and K rows of
// one FullAttention layer before the RoPE kernel rotates them in place.
//
// KVMem's retrieval statistics must see Q/K exactly as the KV writer would
// have (post q_norm/k_norm, pre-RoPE): that is the frame every historical
// K was summarized in. The FA layer body overwrites fa_q_batch/fa_k_batch
// with the RoPE outputs, so this tap is the only point where those rows can
// still be read. It only copies; it never modifies either input, so the
// model arithmetic is bit-identical to the untapped forward.
//
// K: ALL `n_rows` rows, batch order. The engine feeds them to the per-slot
//    KVMem key statistics right after the forward.
// Q: only each slot's LAST row. Rows are packed per slot (m_per_slot), so
//    the last row of a slot is the one whose successor belongs to a
//    different slot — the row whose hidden state the sampler reads, and the
//    query KVMem's retrieval scoring evaluates against historical blocks.
//
// Grid: [ceil(max(q_width,k_width) / 256), n_rows, 2]  (z: 0=K, 1=Q)
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256)
__global__ void kvmem_tap_f32(
    const float* __restrict__ q,       // [n_rows × q_width]
    const float* __restrict__ k,       // [n_rows × k_width]
    float* __restrict__ q_tap,         // [n_slots × q_width]
    float* __restrict__ k_tap,         // [max_rows × k_width]
    const int* __restrict__ row_slot,  // [n_rows] slot id per row
    int n_rows,
    int q_width,
    int k_width,
    int n_slots
) {
    const int row = blockIdx.y;
    if (row >= n_rows) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.z == 0) {
        if (i < k_width) {
            k_tap[(long long)row * k_width + i] = k[(long long)row * k_width + i];
        }
        return;
    }

    if (i >= q_width) return;
    const int slot = row_slot[row];
    if (slot < 0 || slot >= n_slots) return;
    const int last = (row + 1 == n_rows) || (row_slot[row + 1] != slot);
    if (!last) return;
    q_tap[(long long)slot * q_width + i] = q[(long long)row * q_width + i];
}
