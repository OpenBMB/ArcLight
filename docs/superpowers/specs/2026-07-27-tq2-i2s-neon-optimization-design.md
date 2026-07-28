# Design: TQ2_0 I2S NEON Optimization for 158BitNet Decode

- **Date:** 2026-07-27
- **Status:** Approved (pending spec review)
- **Branch:** `optimize-tq2-neon-i2s`
- **Reference:** `/Users/chersu/workdir/AI/158BitNet` (I2S NEON kernels in `src/quant_tq2_0.c`)
- **Predecessor:** `2026-07-27-158bitnet-support-design.md` (added TQ2_0 support; merged to `main`)

## 1. Goal

Accelerate TQ2_0 (ternary) decode on ARM to make 158BitNet models practical. **Acceptance bar: 0.5B decode ≥ 150 tok/s** on this Apple M4 machine (currently 19.6 tok/s with the scalar reference path — a ~7.6× target). The same kernel auto-accelerates the 1B/3B/8B models.

## 2. Why the scalar path is slow

The current `nnml_vec_dot_tq2_0_q8_K` (added in the predecessor branch) dequantizes each 256-element TQ2_0 block to 256 floats on the fly and does an F32 dot product — pure scalar C++. For ternary weights this is the worst-case cost: every weight is unpacked to a float only to be multiplied and discarded. 158BitNet's optimized path instead keeps weights packed and uses ARM `vdotq_s32` (int8 × int8 → int32) over an int8-quantized activation, with a "bsums trick" that avoids materializing the ternary values as floats at all.

158BitNet's default ARM fast path is `BITNET_USE_TQ2_I2S=1`: weights are reordered once at load time into a 4-row-packed "I2S" layout (`bitnet_tq2_0_reorder_to_i2s`), then `bitnet_tq2_0_matmul_i2s_neon_parallel` computes four output rows per pass with shared int8 activation reuse + `vdotq_s32`. The README records >100 tok/s on the slower Snapdragon 865 for the 0.5B; M4 (faster, has ARMv8.2 dotprod) should clear 150.

## 3. Approach

A dedicated TQ2_0 matmul path in `nnml`, mirroring ArcLight's existing Q4_0 `is_asm_gemm` kernel mechanism (`forward_mul_mat<block_q4_0,nr,nc,Q8_0>` → `nnml_gemm_q4_0_*_q8_0` in `nnml/src/ops/arm/gemm.cpp`, plus load-time `format_repack`). On ARM dotprod builds it replaces the scalar `nnml_vec_dot_tq2_0_q8_K` for TQ2_0 weight tensors; the scalar path stays as the non-ARM / non-dotprod fallback.

Three ported pieces + one dispatch wiring:

### 3.1 Load-time I2S reorder (port `bitnet_tq2_0_reorder_to_i2s`)
For each TQ2_0 weight tensor, reorder into the I2S 4-row-packed layout and emit the companion arrays:
- `packed` — 2-bit codes, 4 output rows interleaved per group (layout: `n_groups * sub_blocks_per_group * QK_I2S` bytes; 4 consecutive I2S sub-blocks share one TQ2_0 scale).
- `scales` — per I2S sub-block, 4 floats (one per row).
- `bsums` — per-row block sums (for the activation-side bsum correction).

Built once at model load and cached for the model's lifetime (the analog of 158BitNet's `model->i2s_cache`).

### 3.2 int8 activation quantization (port `bitnet_tq2_0_quantize_vec_i8`)
The RMSNorm output (a row of the activation) → int8 vector + a scalar absmax scale + per-256-element block bsums. NEON-accelerated (`vmaxq_f32` for absmax, vectorized float→int8). This is the format the I2S kernel consumes. It bridges ArcLight's existing activation-quantization step (which currently produces Q8_K for the scalar vec_dot); the int8+bsums layout is compatible with what `quantize_row_q8_K` produces, so this is a thin adapter rather than a second quantization.

### 3.3 I2S NEON matmul (port `bitnet_tq2_0_matmul_i2s_neon_parallel`)
The hot kernel: for each group of 4 output rows, dot the packed weights against the int8 activation using `vdotq_s32`, apply the per-row scales, and subtract the precomputed activation block-bsum correction once per block. Multi-threaded across output-row groups via ArcLight's existing threadpool (the scalar path already parallelizes the same way through `forward_mul_mat_generic`).

### 3.4 Dispatch + weight flag
- Add a TQ2_0 case to the `is_asm_gemm` branch of `nnml_compute_forward_mul_mat` (`nnml/src/ops/ops.cpp`) that calls the new I2S path.
- Set `is_asm_gemm` on TQ2_0 weight tensors at load (`src/model.cpp`), gated on ARM-dotprod availability, so the new path is taken (mirrors how Q4_0 weights get the flag).
- Trigger the I2S reorder (3.1) for each TQ2_0 weight at load.

## 4. Components / files

| File | Change |
| --- | --- |
| `nnml/src/ops/arm/tq2_i2s.cpp` (new) | Port `bitnet_tq2_0_reorder_to_i2s` + `bitnet_tq2_0_quantize_vec_i8` + `bitnet_tq2_0_matmul_i2s_neon_parallel` (NEON, gated `__ARM_NEON` + `__ARM_FEATURE_DOTPROD`). Wired into `nnml/CMakeLists.txt` arm glob. |
| `nnml/src/ops/arm/tq2_i2s.h` (new) | Public decls for the three functions. |
| `nnml/src/ops/ops.cpp` | TQ2_0 case in `nnml_compute_forward_mul_mat` asm branch → calls the I2S path. |
| `nnml/CMakeLists.txt` | (No change expected — arm glob already picks up `src/ops/arm/*.cpp` when `NEON` is on. Confirm.) |
| `src/model.cpp` | Set `is_asm_gemm` on TQ2_0 weights + invoke the I2S reorder at load, gated on ARM dotprod. |
| `nnml/test/test-tq2-0.cpp` | Add a correctness case comparing the I2S kernel output vs the scalar `nnml_vec_dot_tq2_0_q8_K` on a known weight×activation pair. |

Scalar `nnml_vec_dot_tq2_0_q8_K` and its trait registration are unchanged (fallback for non-ARM).

## 5. Verification

- **Correctness:**
  - New unit test: I2S kernel output matches `nnml_vec_dot_tq2_0_q8_K` (within float tolerance) on a hand-built TQ2_0 block × int8 activation pair.
  - End-to-end: 0.5B still decodes "Paris" (coherent); first-token logits match the scalar-path output within tolerance (the scalar path is the correctness oracle, already cross-checked against 158BitNet). 1B/3B/8B regression: still correct.
- **Performance (acceptance bar):** 0.5B decode ≥ 150 tok/s with `--threads 8 --nodes 1 --numa none` on this M4. Measured via `al-gen`'s existing `decode time: … token/s` output. Report 1B/3B/8B numbers too (expected to scale up similarly).
- **No regression on non-TQ2_0 / non-ARM:** the Q4_0/Q4_K/Q6_K paths and the llama-arch models are untouched; on non-ARM builds the scalar TQ2_0 fallback is unchanged.

## 6. Risks / to resolve in the plan

1. **I2S cache storage.** ArcLight weights are single-buffer `nnml_tensor`s; the I2S reorder produces three arrays (packed + scales + bsums) per weight. The plan must decide where these live — most likely a per-weight I2S cache allocated at load and referenced from the tensor (e.g., via a side allocation hung off the model, like 158BitNet's `i2s_cache.blocks[il]`). This is the main integration surface.
2. **Activation quantization bridging.** The I2S kernel wants int8 + scalar scale + per-block bsums; ArcLight's generic mul_mat currently quantizes src1 to the type's `vec_dot_type` (Q8_K for TQ2_0). The plan must wire the I2S path to quantize src1 with `bitnet_tq2_0_quantize_vec_i8` (or confirm Q8_K's layout can be consumed directly) and feed the kernel.
3. **Threading model fit.** 158BitNet parallelizes with its own pthread worker pool; ArcLight parallelizes via `nnml_threadpool` + `nnml_compute_state`. The kernel must chunk output-row groups across `ith/nth` like the existing `forward_mul_mat_generic` does, not spawn its own threads.
4. **Correctness of the bsums trick at the integration boundary** (scale application order, the int8 activation scale folding into the per-row weight scales). Validate against the scalar oracle.
5. **150 tok/s is aggressive.** If the core I2S GEMM lands short, the documented levers are the out-of-scope fusions (fused RMSNorm→int8, fused QKV, gate+up paired, output cache). Defer unless the bar is missed.

## 7. Out of scope

- x86 TQ2_0 kernels (non-ARM keeps the scalar fallback).
- Fused RMSNorm→int8 quant, fused QKV, gate+up paired kernel, Q6_K/F16 output-projection cache (only pulled in if the 0.5B misses 150 tok/s).
- TP (`--numa tp`) tuning of the I2S path (single-node verification only).
