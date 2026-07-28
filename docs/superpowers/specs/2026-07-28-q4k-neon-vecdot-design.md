# Design: NEON Q4_K vec_dot

- **Date:** 2026-07-28
- **Status:** Approved (pending spec review)
- **Branch:** `optimize-q4k-neon-vecdot`
- **Scope:** Single `vec_dot` function — small, focused change.

## 1. Goal & motivation

After the TQ2_0 I2S optimization, every ArcLight quant type has a NEON `vec_dot` **except Q4_K**, whose `nnml_vec_dot_q4_K_q8_K` (`nnml/src/ops/types.cpp:437`) is scalar (dequantize each 256-element block to f32, then an f32 dot). This is the last slow path on ARM. MiniCPM5-1B-Q4_K_M (Q4_K projections) decodes at **10.71 tok/s** on this M4 because of it.

| type | vec_dot location | status |
| --- | --- | --- |
| TQ2_0 | arm/tq2_i2s.cpp | NEON ✅ |
| Q4_0 | arm/gemm.cpp (asm) | NEON ✅ |
| **Q4_K** | **types.cpp:437** | **scalar ❌ (this work)** |
| Q6_K | arm/arm.cpp:748 | NEON ✅ |
| Q8_0 | arm/arm.cpp:1207 | NEON ✅ |
| F16 | arm/arm.cpp:285 | NEON ✅ |

## 2. Approach

Add a NEON `nnml_vec_dot_q4_K_q8_K` in `nnml/src/ops/arm/arm.cpp`, modeled exactly on the existing Q6_K NEON (`arm.cpp:748`), which already proved the pattern. The kernel, per Q4_K block (256 elements = 8 sub-blocks of 32):

1. Extract the 8 6-bit scales + 8 6-bit mins from `block_q4_K.scales[12]` via `get_scale_min_k4` (already in `types.cpp:383`, used by the scalar `dequantize_row_q4_K`).
2. Dot the 4-bit quants (lo/hi nibbles of `qs[]`, raw 0–15, **no** −8 offset — Q4_K differs from Q4_0 here) against the 32-element Q8_K activation using `vdotq_s32` (ARMv8.2 dotprod) — same accumulator pattern as the Q6_K kernel.
3. Apply `d * sc` and subtract `dmin * m` per sub-block; accumulate. (`d`/`dmin` are the super-block fp16 scales.)

The current scalar version becomes the non-NEON fallback, renamed `nnml_vec_dot_q4_K_q8_K_ref` (matching the `_ref`/`_generic` convention used by Q4_0 and Q6_K). The type trait already references the name `nnml_vec_dot_q4_K_q8_K`, which on ARM resolves to the new NEON version and on non-ARM to a fallback. **No dispatch, ops.cpp, or model.cpp changes** — the generic mul_mat already calls the trait's `vec_dot`.

Reference: the canonical ggml/llama.cpp `ggml_vec_dot_q4_K_q8_K` (the same source ArcLight's Q6_K NEON was derived from). 158BitNet has no Q4_K NEON kernel (it uses Q4_K only for embeddings).

## 3. Wiring (mirror Q6_K exactly)

- `nnml/src/ops/arm/arm.cpp`: define `nnml_vec_dot_q4_K_q8_K` (NEON `__ARM_FEATURE_DOTPROD`/MATMUL_INT8 path) + `nnml_vec_dot_q4_K_q8_K_generic` (scalar fallback, the current body) called under `#else`. Same structure as q6_K at arm.cpp:748–803.
- `nnml/src/ops/types.cpp`: rename the current `nnml_vec_dot_q4_K_q8_K` (line 437) to `nnml_vec_dot_q4_K_q8_K_ref` (portable scalar; kept for non-ARM).
- `nnml/src/ops/x86/x86.cpp`: add a thin `nnml_vec_dot_q4_K_q8_K` that calls `_ref` (so x86 still links with the scalar path; a real x86 SIMD version is out of scope). Matches how q6_K has an x86.cpp definition.
- The trait at `types.cpp:60` is unchanged (still references `nnml_vec_dot_q4_K_q8_K`).

## 4. Verification

- **Unit test** (extend `nnml/test/test-tq2-0.cpp` or a new `test-q4k.cpp`): NEON `nnml_vec_dot_q4_K_q8_K` == `_ref` scalar within tolerance on a hand-built Q4_K block × Q8_K activation (reuse `quantize_row_q8_K` for the activation). Register in `nnml/CMakeLists.txt` if a new test file.
- **End-to-end:** MiniCPM5-1B-Q4_K_M (`./tmp/MiniCPM5-1B-Q4_K_M.gguf`) decodes coherently (greedy tokens match the pre-change scalar-path output for the first ~8 tokens), with a **big speedup** (expected ~5–10× over 10.71 tok/s → roughly 50–90 tok/s). No hard bar (per decision); correctness + a large gain is the gate.
- **Regression:** the four bitcpm4 models (Q4_K embeddings) still run correctly.

## 5. Risks

1. **Q4_K block layout / scale extraction correctness.** The 6-bit scale/min packing (`get_scale_min_k4`) and the 4-bit quant nibble layout must match the scalar dequantize exactly. The unit test (NEON == `_ref`) is the backstop.
2. **Link wiring** (two definitions of the same name across TUs). Must rename the types.cpp scalar to `_ref` and ensure arm.cpp (ARM) / x86.cpp (x86) provide the `nnml_vec_dot_q4_K_q8_K` symbol; verify the ARM build links and non-ARM still has a definition.

## 6. Out of scope

- x86 SIMD Q4_K (x86 keeps the scalar via `_ref`).
- The 2-row (`nrc==2`) MATMUL_INT8 path for Q4_K (Q6_K has it; Q4_K can add it later — diminishing returns after the single-row path).
