# TQ2_0 I2S NEON Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 158BitNet's I2S TQ2_0 NEON kernel into ArcLight so 0.5B decode reaches ≥150 tok/s (currently 19.6 tok/s scalar). Auto-accelerates 1B/3B/8B.

**Architecture:** Add a dedicated TQ2_0 matmul path that mirrors ArcLight's existing Q4_0 `is_asm_gemm` mechanism. At model load, each TQ2_0 weight is reordered once into the I2S 4-row-packed layout (cached on the tensor via its unused `padding[8]` slot). At compute time, when a TQ2_0 weight has `is_asm_gemm` set, `nnml_compute_forward_mul_mat` calls a new `forward_mul_mat_tq2_0_i2s` that quantizes the activation to int8 (NEON) and runs the ported `vdotq_s32` I2S kernel. The scalar `nnml_vec_dot_tq2_0_q8_K` remains the fallback for non-ARM / non-dotprod builds.

**Tech Stack:** C++17, ARM NEON (`__ARM_NEON` + `__ARM_FEATURE_DOTPROD`), CMake. Reference kernels: `/Users/chersu/workdir/AI/158BitNet/src/quant_tq2_0.c`.

## Global Constraints

- **C++17**, 4-space indent, braces on same line, `snake_case`; no formatter — no unrelated formatting churn.
- **ARM-only optimization.** All NEON code is gated `#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)`. Non-ARM builds must still compile and use the scalar `nnml_vec_dot_tq2_0_q8_K` unchanged. This M4 has dotprod (confirmed: the NEON build defines `__ARM_FEATURE_DOTPROD`).
- **Correctness oracle:** the existing scalar `nnml_vec_dot_tq2_0_q8_K` (verified against 158BitNet in the predecessor branch). Every I2S output must match it within float tolerance. End-to-end: the 0.5B must still decode "Paris" with first-token logits matching the scalar path.
- **Acceptance bar:** 0.5B decode ≥ 150 tok/s (`--threads 8 --nodes 1 --numa none` on this M4), measured by `al-gen`'s `decode time: … token/s`.
- The I2S kernels are PORTED (transcribed + adapted) from `158BitNet/src/quant_tq2_0.c` — port the named functions, do not reimplement from scratch.
- Branch `optimize-tq2-neon-i2s`. Commit per task. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Models at `/Users/chersu/workdir/AI/158BitNet/models/` — never commit them.

---

## Reference: 158BitNet functions to port (file: `/Users/chersu/workdir/AI/158BitNet/src/quant_tq2_0.c`)

| ArcLight name | Port from (158BitNet) | Notes |
| --- | --- | --- |
| `tq2_i2s_packed_size` | `bitnet_tq2_0_i2s_packed_size` (≈ line 1440) | Layout math. `QK_I2S=64`, `BITNET_TQ2_0_QK=256`. |
| `tq2_i2s_reorder` | `bitnet_tq2_0_reorder_to_i2s` (≈ line 1461) | Produces `packed` + `packed_scales` + `packed_bsums`. Port the full body including the NEON inner pack. |
| `tq2_quantize_vec_i8` | `bitnet_tq2_0_quantize_vec_i8_impl` | NEON absmax + float→int8. Produces `qvec` + `scale` + `block_bsums`. |
| `tq2_matmul_i2s_neon` | `bitnet_tq2_0_matmul_i2s_neon_parallel_impl` (and the non-parallel `bitnet_tq2_0_matmul_i2s_neon` ≈ line 1781) | The hot `vdotq_s32` kernel. Port the per-row-group body; ArcLight drives parallelism via `ith/nth` (see Task 3). |

`BITNET_TQ2_0_QK` = 256 = ArcLight `QK_K`. `BITNET_TQ2_0_QS_SIZE` = 64. `bitnet_fp16_to_fp32` → use ArcLight's `NNML_FP16_TO_FP32`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `nnml/src/ops/arm/tq2_i2s.h` (new) | Public decls: cache struct, `tq2_i2s_packed_size`, `tq2_i2s_build_cache`, `tq2_i2s_free_cache`, `tq2_quantize_vec_i8`, `tq2_matmul_i2s_neon`. |
| `nnml/src/ops/arm/tq2_i2s.cpp` (new) | The four ported functions (NEON, gated). Auto-globbed by `nnml/CMakeLists.txt` arm glob when `NEON` is on. |
| `nnml/src/ops/ops.cpp` | New `forward_mul_mat_tq2_0_i2s` + TQ2_0 case in `nnml_compute_forward_mul_mat` asm branch. |
| `nnml/include/ops.h` | Declaration of `nnml_compute_forward_mul_mat_tq2_0_i2s`. |
| `nnml/include/tensor.h` | `set_i2s_cache` / `get_i2s_cache` accessors (store pointer in `padding[8]`). |
| `src/model.cpp` | At load: for TQ2_0 weights on ARM-dotprod, build the I2S cache + `set_asm_gemm()`. |
| `nnml/test/test-tq2-0.cpp` | Add correctness cases (I2S matmul vs scalar vec_dot). |

The scalar `nnml_vec_dot_tq2_0_q8_K` and its type-trait registration are unchanged.

---

## Task 1: Port I2S reorder + int8 quantization primitives

**Files:**
- Create: `nnml/src/ops/arm/tq2_i2s.h`, `nnml/src/ops/arm/tq2_i2s.cpp`
- Modify: `nnml/test/test-tq2-0.cpp`

**Interfaces:**
- Produces:
  ```cpp
  // tq2_i2s.h
  #pragma once
  #include "../ops.h"          // for block_tq2_0, QK_K, nnml_fp16_t, NNML_FP16_TO_FP32
  struct tq2_i2s_cache {
      uint8_t * packed;        // 4-row-packed 2-bit codes
      float   * scales;        // per I2S sub-block, 4 floats (one per row)
      int32_t * bsums;         // per I2S sub-block, 4 int32 (one per row)
      int      out_dim;
      int      in_dim;
  };
  size_t tq2_i2s_packed_size(int out_dim, int in_dim);
  // Allocate + fill a cache (caller frees with tq2_i2s_free_cache). Returns nullptr on bad args / non-ARM.
  tq2_i2s_cache * tq2_i2s_build_cache(const void * tq2_weight, int out_dim, int in_dim);
  void   tq2_i2s_free_cache(tq2_i2s_cache * c);
  // Activation quant: f32 row -> int8 + scalar absmax scale + per-256 block bsums. Returns 0 on success.
  int    tq2_quantize_vec_i8(const float * vec, int n, int8_t * qvec, float * scale, int32_t * block_bsums);
  ```
- The reorder/quant bodies are **ported verbatim** from `bitnet_tq2_0_reorder_to_i2s` and `bitnet_tq2_0_quantize_vec_i8_impl` (see reference table), adapting: `BITNET_TQ2_0_QK`→`QK_K`, `BITNET_TQ2_0_QS_SIZE`→`QK_K/4` (64), `bitnet_fp16_to_fp32`→`NNML_FP16_TO_FP32`. Wrap the whole file body in `#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)` … `#else` stubs that return nullptr/-1 `#endif` so non-ARM builds link (the dispatch never calls them there).

- [ ] **Step 1: Write failing test (reorder round-trip)**

Append to `nnml/test/test-tq2-0.cpp` (after the existing vec_dot cases, before the final pass/fail print), guarded so the binary still builds on non-ARM:
```cpp
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include "../src/ops/arm/tq2_i2s.h"
static void check_i2s_cache(void) {
    // 4 output rows x 256-element row (one TQ2_0 block per row). All weights +1 (qs=0xAA, d=1.0).
    const int out_dim = 4, in_dim = QK_K;
    block_tq2_0 w[4];
    for (int r = 0; r < 4; ++r) { w[r].d = NNML_FP32_TO_FP16(1.0f); for (int i=0;i<QK_K/4;++i) w[r].qs[i] = 0xAA; }
    tq2_i2s_cache * c = tq2_i2s_build_cache(w, out_dim, in_dim);
    NNML_ASSERT(c != nullptr);
    // packed_size for 1 group, in_dim/64 = 4 sub-blocks -> 4*64 = 256 bytes
    NNML_ASSERT(tq2_i2s_packed_size(out_dim, in_dim) == 256);
    // Each row's scale d == 1.0
    NNML_ASSERT(fabsf(c->scales[0] - 1.0f) < 1e-6f);
    tq2_i2s_free_cache(c);
}
#endif
```
Call `check_i2s_cache();` inside `main` (inside the ARM guard) before the final pass/fail print.

- [ ] **Step 2: Build the test, confirm it fails to link**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -5
```
Expected: link error — `tq2_i2s_*` undefined (not implemented yet).

- [ ] **Step 3: Implement `tq2_i2s.h` + `tq2_i2s.cpp`**

Create `tq2_i2s.h` with the interface above. Create `tq2_i2s.cpp`:
- `#include "../src/ops/arm/tq2_i2s.h"` and `<cstdlib>`.
- Wrap everything in `#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)`.
- Port `tq2_i2s_packed_size` ← `bitnet_tq2_0_i2s_packed_size` (replace `BITNET_TQ2_0_QK`→`QK_K`, `QK_I2S`→`64` local const).
- Port `tq2_i2s_build_cache`: `malloc` packed/scales/bsums (sizes from `tq2_i2s_packed_size` + `n_groups*blocks_per_row*4` floats + `n_groups*sub_blocks*4` int32), then call the ported reorder body ← `bitnet_tq2_0_reorder_to_i2s` (adapt macros per the reference table). Return a heap `tq2_i2s_cache*`.
- Port `tq2_quantize_vec_i8` ← `bitnet_tq2_0_quantize_vec_i8_impl`.
- `tq2_i2s_free_cache`: `free` the three buffers + the struct.
- `#else` (non-ARM): stub all four to return `0`/`nullptr`/`-1`.

- [ ] **Step 4: Build + run the test, confirm PASS**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -3
./build/nnml/test-tq2-0
```
Expected: `test-tq2-0: PASS`.

- [ ] **Step 5: Confirm the full nnml build is clean**

```sh
cmake --build build -j 2>&1 | tail -10
```
Expected: success (the new file is auto-globbed by the arm glob; non-ARM stubs keep it portable).

- [ ] **Step 6: Commit**

```sh
git add nnml/src/ops/arm/tq2_i2s.h nnml/src/ops/arm/tq2_i2s.cpp nnml/test/test-tq2-0.cpp
git commit -m "Port TQ2_0 I2S reorder + int8 quantization primitives (ARM NEON)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Port the I2S NEON matmul kernel + correctness test vs scalar

**Files:**
- Modify: `nnml/src/ops/arm/tq2_i2s.h`, `nnml/src/ops/arm/tq2_i2s.cpp`
- Modify: `nnml/test/test-tq2-0.cpp`

**Interfaces:**
- Produces (add to `tq2_i2s.h`):
  ```cpp
  // Dot one I2S weight row-group against an int8 activation row.
  //   c        : built cache (out_dim rows, in_dim)
  //   qvec     : int8 activation, in_dim elements
  //   vec_scale: activation absmax scale (from tq2_quantize_vec_i8)
  //   bsums    : per-256 activation block sums (from tq2_quantize_vec_i8), in_dim/256 ints
  //   out      : output floats, out_dim elements (WRITTEN, not accumulated)
  void tq2_matmul_i2s_neon(const tq2_i2s_cache * c, const int8_t * qvec, float vec_scale,
                           const int32_t * bsums, float * out);
  ```
- Port the body ← `bitnet_tq2_0_matmul_i2s_neon` (≈ line 1781; the single-pass, non-pthread body), which does the `vdotq_s32` accumulation + per-row scale + bsum correction. Do NOT port the pthread-parallel wrapper — ArcLight drives parallelism externally (Task 3).

- [ ] **Step 1: Write failing test (I2S matmul == scalar vec_dot)**

Append to `nnml/test/test-tq2-0.cpp` inside the ARM guard:
```cpp
static void check_i2s_vs_scalar(void) {
    // 4 output rows, in_dim = QK_K. Build weights with mixed ternary codes.
    const int out_dim = 4, in_dim = QK_K;
    block_tq2_0 w[4];
    // row0 all +1 (0xAA), row1 all -1 (0x00), row2 all 0 (0x55 -> code 1), row3 mixed
    for (int i=0;i<QK_K/4;++i){ w[0].qs[i]=0xAA; w[1].qs[i]=0x00; w[2].qs[i]=0x55; w[3].qs[i]=0x96; }
    for (int r=0;r<4;++r) w[r].d = NNML_FP32_TO_FP16(1.0f);
    tq2_i2s_cache * c = tq2_i2s_build_cache(w, out_dim, in_dim);

    // activation = all 1.0
    float act[QK_K]; for (int i=0;i<QK_K;++i) act[i]=1.0f;
    int8_t qvec[QK_K]; float vscale; int32_t bsums[QK_K/256];
    tq2_quantize_vec_i8(act, QK_K, qvec, &vscale, bsums);

    float out_i2s[4];
    tq2_matmul_i2s_neon(c, qvec, vscale, bsums, out_i2s);

    // scalar oracle: nnml_vec_dot_tq2_0_q8_K expects a block_q8_K activation.
    // Build one from act (all 1.0) for direct comparison.
    block_q8_K y; quantize_row_q8_K(act, &y, QK_K);
    for (int r=0;r<4;++r){
        float s_scalar = 0.0f;
        nnml_vec_dot_tq2_0_q8_K(QK_K, &s_scalar, 0, &w[r], 0, &y, 0, 1);
        // I2S dot is row r; allow tolerance for int8 quant rounding.
        if (fabsf(out_i2s[r] - s_scalar) > 0.05f * (fabsf(s_scalar)+1.0f)) {
            printf("FAIL i2s row %d: i2s=%g scalar=%g\n", r, out_i2s[r], s_scalar); ++g_fails;
        }
    }
    tq2_i2s_free_cache(c);
}
```
Call `check_i2s_vs_scalar();` in `main` (ARM guard) before the final print.

- [ ] **Step 2: Build, confirm it fails to link**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -5
```
Expected: `tq2_matmul_i2s_neon` undefined.

- [ ] **Step 3: Implement `tq2_matmul_i2s_neon`**

In `tq2_i2s.cpp`, port the body ← `bitnet_tq2_0_matmul_i2s_neon` (the non-parallel variant). Adapt: read packed/scales/bsums from the `tq2_i2s_cache*`; iterate the 4 rows × sub-blocks; use `vdotq_s32` (+ the bsum correction) exactly as the reference. Add `void tq2_matmul_i2s_neon(...)` to the header.

- [ ] **Step 4: Build + run, confirm PASS (I2S matches scalar within tolerance)**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -3
./build/nnml/test-tq2-0
```
Expected: `test-tq2-0: PASS`. If a row mismatches by more than 5%, the bsum-correction or per-row-scale application is wrong — re-check against `bitnet_tq2_0_matmul_i2s_neon`.

- [ ] **Step 5: Commit**

```sh
git add nnml/src/ops/arm/tq2_i2s.h nnml/src/ops/arm/tq2_i2s.cpp nnml/test/test-tq2-0.cpp
git commit -m "Port TQ2_0 I2S NEON matmul kernel (vdotq_s32); matches scalar vec_dot

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire the I2S path into ArcLight's matmul + load

**Files:**
- Modify: `nnml/include/tensor.h` (i2s_cache accessors), `nnml/include/ops.h` (decl), `nnml/src/ops/ops.cpp` (dispatch + `forward_mul_mat_tq2_0_i2s`), `src/model.cpp` (load-time cache build + flag)

**Interfaces:**
- Consumes: Task 1/2 `tq2_i2s_cache`, `tq2_i2s_build_cache`, `tq2_quantize_vec_i8`, `tq2_matmul_i2s_neon`.
- Produces:
  - `nnml_tensor::set_i2s_cache(void*)` / `get_i2s_cache()` (store the pointer in `padding[8]`).
  - `void nnml_compute_forward_mul_mat_tq2_0_i2s(nnml_tensor * node, const nnml_compute_state * params);`

- [ ] **Step 1: Add i2s_cache accessors to `nnml_tensor`**

In `nnml/include/tensor.h`, the struct has `char padding[8];` (line ~184). Add accessors next to `is_asm_gemm` (line ~153):
```cpp
    void * get_i2s_cache() const noexcept { void * p; memcpy(&p, padding, sizeof(p)); return p; }
    void   set_i2s_cache(void * p) noexcept { memcpy(padding, &p, sizeof(p)); }
```
Use `memcpy` (not a cast) to avoid alignment UB. (If a build error shows `padding` is not 8 bytes or not pointer-aligned, assert `NNML_ALIGN_UP(sizeof(nnml_tensor)) == NNML_TENSOR_SIZE` is unchanged — it is, since `padding[8]` is replaced byte-for-byte. The struct size does not change.)

- [ ] **Step 2: Implement `nnml_compute_forward_mul_mat_tq2_0_i2s`**

In `nnml/src/ops/ops.cpp`, add (after `nnml_compute_forward_mul_mat`):
```cpp
#include "ops/arm/tq2_i2s.h"   // at top of file with other includes

void nnml_compute_forward_mul_mat_tq2_0_i2s(nnml_tensor * node, const nnml_compute_state * params) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const nnml_tensor * src0 = node->get_src_tensor(0);  // weight (TQ2_0)
    const nnml_tensor * src1 = node->get_src_tensor(1);  // activation (F32)

    NNML_TENSOR_BINARY_OP_LOCALS   // ne0..ne3, nb0..nb3, ne00.., nb00..
    const tq2_i2s_cache * c = (const tq2_i2s_cache *) src0->get_i2s_cache();
    NNML_ASSERT(c != nullptr);

    const int out_dim = ne0;          // = ne01 (weight rows)
    const int in_dim  = ne00;         // contraction dim (must be multiple of QK_K)
    NNML_ASSERT(in_dim % QK_K == 0);

    const int ith = params->ith, nth = params->nth;
    // distribute output rows across threads in groups of 4 (I2S processes 4 rows per call)
    const int n_groups = (out_dim + 3) / 4;
    const int dr = (n_groups + nth - 1) / nth;
    const int g0 = dr * ith;
    const int g1 = (g0 + dr < n_groups) ? (g0 + dr) : n_groups;

    // per-thread int8 activation scratch (one row of in_dim + bsums)
    int8_t * qvec = (int8_t *) params->work_data + ith * in_dim;
    int32_t * bsums = (int32_t *)(qvec + in_dim);   // in_dim/256 ints; ensure work buffer is large enough (see Step 5 note)

    // src1 layout: [in_dim, n_tokens] column-major-ish; iterate the n_tokens columns (ne1).
    // For decode ne1 == 1 (single token). Generalize over ne1 for prefill.
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        const float * act_row = (const float *)((char *)src1->tensor_data() + i1*nb1);
        float vscale;
        tq2_quantize_vec_i8(act_row, in_dim, qvec, &vscale, bsums);
        for (int g = g0; g < g1; ++g) {
            float out4[4];
            const tq2_i2s_cache cgrp = *c;   // shallow copy; offset packed/scales/bsums to group g
            // NOTE: tq2_matmul_i2s_neon as ported computes ALL out_dim rows for the full cache.
            //   If your Task-2 port already iterates all groups internally, call it once here and
            //   write into node row i1's output; do NOT loop g. Pick ONE convention and keep
            //   Task 2's test consistent with it. (See Step 4 verification.)
            tq2_matmul_i2s_neon(&cgrp, qvec, vscale, bsums, out4);
            float * dst_row = (float *)((char *)node->tensor_data() + g*4*nb0 + i1*nb1);
            for (int r = 0; r < 4 && (g*4 + r) < out_dim; ++r) dst_row[r*nb0/sizeof(float)] = out4[r];
        }
    }
#else
    NNML_ABORT("TQ2_0 I2S path built on non-ARM/non-dotprod — should not be reached");
#endif
}
```
**Important convention note (resolve in implementation):** decide whether `tq2_matmul_i2s_neon` computes a single 4-row group (then the `g` loop above stands) or all `out_dim` rows at once (then drop the `g` loop and write the full output). The Task-2 test must match this convention. The simplest is: have `tq2_matmul_i2s_neon` take a group index `g` and compute that one 4-row group; then the loop here parallelizes across `g` via `ith/nth`. Adjust the Task-2 signature/comment to match (`tq2_matmul_i2s_neon(c, g, qvec, vscale, bsums, out4)`).

- [ ] **Step 3: Declare + dispatch**

- In `nnml/include/ops.h`, near the other `nnml_compute_forward_*` decls: `void nnml_compute_forward_mul_mat_tq2_0_i2s(nnml_tensor * node, const nnml_compute_state * params);`
- In `nnml/src/ops/ops.cpp` `nnml_compute_forward_mul_mat`, change the asm guard (≈ line 1304) from:
  ```cpp
  if (node->is_asm_gemm() && src0->get_data_type() == NNML_TYPE_Q4_0) {
  ```
  to:
  ```cpp
  if (node->is_asm_gemm()) {
      if (src0->get_data_type() == NNML_TYPE_TQ2_0) {
          nnml_compute_forward_mul_mat_tq2_0_i2s(node, params);
          return;
      }
      if (src0->get_data_type() != NNML_TYPE_Q4_0) {
          NNML_ABORT("asm_gemm: unsupported weight type %d", (int) src0->get_data_type());
      }
  }
  ```
  (so the Q4_0 block below only runs for Q4_0, and TQ2_0 routes to the I2S path.)

- [ ] **Step 4: Build the I2S cache at load + set the flag**

In `src/model.cpp`, after the existing `format_repack(weight_tensor)` call (≈ line 394) and after `set_data_type`/`is_asm_gemm` handling, add (guarded for ARM dotprod):
```cpp
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (weight_tensor->get_data_type() == NNML_TYPE_TQ2_0) {
        int64_t ne0 = weight_tensor->get_elements(0);   // in_dim (contraction)
        int64_t ne1 = weight_tensor->get_elements(1);   // out_dim
        // weight is stored [in_dim, out_dim] (ggml mul_mat src0 convention) — confirm via the
        // existing Q4_0 path which uses the same tensor; in_dim=ne[0], out_dim=ne[1].
        tq2_i2s_cache * c = tq2_i2s_build_cache(weight_tensor->tensor_data(), (int)ne1, (int)ne0);
        if (c) { weight_tensor->set_i2s_cache(c); weight_tensor->set_asm_gemm(); }
    }
#endif
```
Add `#include "ops/arm/tq2_i2s.h"` to `src/model.cpp`. (The cache is malloc'd by `tq2_i2s_build_cache`; it lives for the model's lifetime. Add a cleanup pass in the model destructor if/when added — out of scope for this task, note as follow-up.)

- [ ] **Step 5: Build + smoke-test (correctness via the live model)**

```sh
cmake --build build -j 2>&1 | tail -15
```
Expected: clean build. If the per-thread work buffer is too small for `qvec+bsums` (in_dim + in_dim/256*4 bytes per thread), bump the work buffer: the `--work_gb` arg controls it; default 2 GB is plenty for one row. If you see corruption, double-check the `params->work_data` offset arithmetic.

Then sanity-run the 0.5B (correctness — must still decode "Paris"):
```sh
./build/al-gen --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-0.5b-tq2_0.gguf \
  --prompt "The capital of France is" --threads 8 --nodes 1 --numa none --max_gen 16 2>&1 | grep -A2 "^> "
```
Expected: "Paris." (If garbage: the per-row output write stride (`nb0`) or the in_dim/out_dim convention is wrong — re-check against the Q4_0 path's tensor shape usage.)

- [ ] **Step 6: Commit**

```sh
git add nnml/include/tensor.h nnml/include/ops.h nnml/src/ops/ops.cpp src/model.cpp
git commit -m "Wire TQ2_0 I2S NEON path into matmul dispatch + load-time reorder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Verify correctness + hit the 150 tok/s bar

**Files:** none modified — verification + a regression guard.

- [ ] **Step 1: Correctness — logits match the scalar path**

Temporarily force the scalar path (run with `--asm 0`, which leaves `is_asm_gemm` unset so TQ2_0 falls through to `nnml_vec_dot_tq2_0_q8_K`), capture the 0.5B first-token greedy output; then run the I2S path (`--asm 1`, default). They must produce the same tokens for the first ~8 generated tokens (the int8 activation quant introduces tiny rounding, so greedy tokens should agree for a normal prompt).
```sh
# scalar path
./build/al-gen --model …/bitcpm4-0.5b-tq2_0.gguf --prompt "The capital of France is" --asm 0 --threads 8 --nodes 1 --numa none --max_gen 12 2>&1 | grep -A1 "^> "
# I2S path (default --asm 1)
./build/al-gen --model …/bitcpm4-0.5b-tq2_0.gguf --prompt "The capital of France is" --threads 8 --nodes 1 --numa none --max_gen 12 2>&1 | grep -A1 "^> "
```
Expected: both decode "Paris" (coherent, agreeing tokens). Record both outputs.

- [ ] **Step 2: Performance — 0.5B decode ≥ 150 tok/s (acceptance bar)**

```sh
./build/al-gen --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-0.5b-tq2_0.gguf \
  --prompt "The capital of France is" --threads 8 --nodes 1 --numa none --max_gen 64 2>&1 | grep "token/s"
```
Expected: `decode time: … <X> token/s` with **X ≥ 150**. If X < 150, capture the number and the breakdown; the levers (in priority order) are: (a) confirm dotprod is actually compiled in (`__ARM_FEATURE_DOTPROD` — check the build defines), (b) confirm the kernel is the I2S 4-row path not a fallback, (c) thread sweep 4/6/8 (8 was optimal on M4 for the scalar path), (d) only then consider the out-of-scope fusions.

- [ ] **Step 3: Regression — 1B/3B/8B still correct and faster**

```sh
for sz in 1b 3b 8b; do
  ./build/al-gen --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-$sz-tq2_0.gguf \
    --prompt "The capital of France is" --threads 8 --nodes 1 --numa none --max_gen 24 2>&1 | grep -E "^> |token/s" | tr '\n' ' ' ; echo "  ($sz)"
done
```
Expected: all three decode "Paris" coherently, with decode tok/s noticeably higher than the scalar baseline (19.6/6.45/2.70/1.18 → record new numbers).

- [ ] **Step 4: Commit a verification note**

```sh
git commit --allow-empty -m "Verify TQ2_0 I2S: 0.5B decode=<X> tok/s (bar 150); 1B/3B/8B correct+ faster

0.5B scalar(--asm 0): '<output>'   I2S: '<output>'
1B/3B/8B decode tok/s: <numbers>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(substitute the real numbers/outputs.)

---

## Self-Review (completed)

**Spec coverage:** spec §3.1 (reorder) → Task 1; §3.2 (int8 quant) → Task 1; §3.3 (matmul kernel) → Task 2; §3.4 (dispatch + flag) → Task 3; §5 (correctness + 150 tok/s bar) → Tasks 2 & 4; §6 risks (cache storage → Task 3 Step 1 `padding[8]`; activation bridging → Task 3 Step 2; threading → Task 3 Step 2 `ith/nth`; bsums correctness → Task 2 oracle test; 150-miss levers → Task 4 Step 2). ✓

**Placeholder scan:** the kernel bodies are referenced as ports from named 158BitNet functions (the established transcription pattern that worked for the scalar dequant). The ArcLight glue (accessors, dispatch, load hook, `forward_mul_mat_tq2_0_i2s`) is written out in full. The one genuine ambiguity — whether `tq2_matmul_i2s_neon` computes one 4-row group or all rows — is called out with a single convention to pick (Task 3 Step 2 note), not left as TBD.

**Type consistency:** `tq2_i2s_cache`, `tq2_i2s_build_cache/free`, `tq2_quantize_vec_i8`, `tq2_matmul_i2s_neon`, `set_i2s_cache/get_i2s_cache`, `nnml_compute_forward_mul_mat_tq2_0_i2s` — names consistent across tasks.
