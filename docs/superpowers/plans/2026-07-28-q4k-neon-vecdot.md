# NEON Q4_K vec_dot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (or executing-plans). Steps use `- [ ]` checkboxes.

**Goal:** Replace the scalar `nnml_vec_dot_q4_K_q8_K` with a NEON dotprod version (the last scalar `vec_dot` on ARM), accelerating all Q4_K models — MiniCPM5-1B-Q4_K_M (10.71 tok/s) and the bitcpm4 Q4_K embeddings. No hard tok/s bar; correctness + a big speedup.

**Architecture:** Add `nnml_vec_dot_q4_K_q8_K` (NEON) to `nnml/src/ops/arm/arm.cpp`, modeled on the existing Q6_K NEON (`arm.cpp:748`). Rename the current scalar to `_ref` (portable fallback); add an x86.cpp passthrough. The type trait already references the name — no dispatch/load changes.

**Tech Stack:** C++17, ARM NEON dotprod (`__ARM_FEATURE_DOTPROD` / `__ARM_FEATURE_MATMUL_INT8`).

## Global Constraints

- **C++17**, 4-space indent, braces same line, snake_case; no formatter — no unrelated formatting churn.
- **ARM NEON** kernel gated on `__ARM_FEATURE_DOTPROD` (single-row path) / `__ARM_FEATURE_MATMUL_INT8` (optional 2-row). Non-ARM builds must still compile and use the scalar `_ref`.
- **Correctness oracle:** the existing scalar `nnml_vec_dot_q4_K_q8_K` (now `_ref`). NEON output must match within float tolerance.
- Branch `optimize-q4k-neon-vecdot`. Commit per task. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Model at `./tmp/MiniCPM5-1B-Q4_K_M.gguf` — do not commit.

## The Q4_K dot math (per 256-element block = 8 sub-blocks of 32)

`block_q4_K` = `{ half d; half dmin; uint8_t scales[12]; uint8_t qs[QK_K/2=128]; }`. `block_q8_K` = `{ float d; int8_t qs[256]; int16_t bsums[16]; }`.

For sub-block `j` (0..7), via `get_scale_min_k4(j, scales, &sc, &m)` (in `types.cpp:383`): `d_sub = NNML_FP16_TO_FP32(d) * sc`, `min_sub = NNML_FP16_TO_FP32(dmin) * m`. The 32 quants are the lo/hi nibbles of `qs[]` (raw 0–15, **no** −8 offset). Per the scalar `dequantize_row_q4_K` (`types.cpp:397`): `weight = d_sub * nibble − min_sub`. So the dot for one sub-block (32 elements) against a Q8_K activation block with scale `y_d`:

```
sum_nq = dot_i8(nibble[0..31], y_qs[0..31])     // vdotq_s32; nibbles are 0..15 → int8
sum_y  = sum_i8(y_qs[0..31])                      // or from bsums
contrib = y_d * (d_sub * sum_nq − min_sub * sum_y)
```

Sum `contrib` over the 8 sub-blocks and over all `nb = n/QK_K` blocks. This is the Q6_K NEON kernel's shape (dequant-to-int8 → `vdotq_s32` → apply scales), just with Q4_K's nibble+scale/min extraction instead of Q6_K's ql/qh.

## Reference templates (local — read these before writing the kernel)

- **Structural template:** `nnml/src/ops/arm/arm.cpp:748` `nnml_vec_dot_q6_K_q8_K` — the NEON dequant+vdotq loop, the `_generic` fallback under `#else`, and (under `MATMUL_INT8`) the `nrc==2` 2-row variant. Model the Q4_K single-row (`nrc==1`) path on this.
- **Q4_K scale/min extraction:** `get_scale_min_k4` (`types.cpp:383`) and `dequantize_row_q4_K` (`types.cpp:397`) — show exactly how scales/mins/nibbles map.
- **Scalar oracle (to rename → `_ref`):** `nnml_vec_dot_q4_K_q8_K` (`types.cpp:437`).
- **x86 passthrough pattern:** `nnml/src/ops/x86/x86.cpp:762` `nnml_vec_dot_q6_K_q8_K` (x86 has its own impl; for Q4_K, a thin wrapper calling `_ref` is acceptable since x86 SIMD is out of scope).

---

## Task 1: NEON Q4_K vec_dot + scalar `_ref` fallback + unit test

**Files:**
- Modify: `nnml/src/ops/arm/arm.cpp` (add NEON `nnml_vec_dot_q4_K_q8_K` + `nnml_vec_dot_q4_K_q8_K_generic`)
- Modify: `nnml/src/ops/types.cpp` (rename scalar → `nnml_vec_dot_q4_K_q8_K_ref`)
- Modify: `nnml/include/ops.h` (declare the new names)
- Modify: `nnml/src/ops/x86/x86.cpp` (passthrough `nnml_vec_dot_q4_K_q8_K` → `_ref`)
- Modify: `nnml/test/test-tq2-0.cpp` (add a Q4_K NEON-vs-scalar case) — or a new `nnml/test/test-q4k.cpp` registered in `nnml/CMakeLists.txt` (implementer's choice; extending the existing test is simpler)

**Interfaces:**
- Consumes: `block_q4_K`, `block_q8_K`, `get_scale_min_k4`, `quantize_row_q8_K`, `QK_K`, `NNML_FP16_TO_FP32`.
- Produces: `nnml_vec_dot_q4_K_q8_K` (NEON, arm.cpp), `nnml_vec_dot_q4_K_q8_K_ref` (scalar, types.cpp — the current body renamed), `nnml_vec_dot_q4_K_q8_K_generic` (arm.cpp scalar fallback that calls `_ref`). The trait at `types.cpp:60` is unchanged.

- [ ] **Step 1: Write the failing test (NEON == scalar oracle)**

Add to `nnml/test/test-tq2-0.cpp` (or a new test file). Construct a Q4_K block with known scales/mins/nibbles, a known activation, and compare the NEON dot to `_ref`:

```cpp
#include "ops.h"   // already included
// build a single Q4_K block (256 weights) + a q8_K activation, compare NEON vs _ref.
static void check_q4k_vecdot(void) {
    block_q4_K w; memset(&w, 0, sizeof(w));
    w.d   = NNML_FP32_TO_FP16(1.0f);
    w.dmin= NNML_FP32_TO_FP16(0.1f);
    // 8 sub-block scales/mins: set scales[] so get_scale_min_k4 returns varied sc/m.
    // Simplest deterministic: scales all 0x01 (sc=1, m=1 for j<4; see get_scale_min_k4).
    for (int i = 0; i < 12; ++i) w.scales[i] = (i % 2) ? 0x21 : 0x12;  // varied 6-bit sc/m
    // qs nibbles: a repeating pattern (0..15)
    for (int i = 0; i < QK_K/2; ++i) w.qs[i] = (uint8_t)((i*7) & 0xFF);

    // activation = a known f32 vector -> q8_K
    float act[QK_K];
    for (int i = 0; i < QK_K; ++i) act[i] = 0.5f * (float)((i % 7) - 3);  // small varied values
    block_q8_K y; quantize_row_q8_K(act, &y, QK_K);

    float s_neon = 0.0f, s_ref = 0.0f;
    nnml_vec_dot_q4_K_q8_K(QK_K, &s_neon, 0, &w, 0, &y, 0, 1);
    nnml_vec_dot_q4_K_q8_K_ref(QK_K, &s_ref, 0, &w, 0, &y, 0, 1);
    // NEON must match the scalar oracle within relative tolerance.
    float tol = 1e-3f * (fabsf(s_ref) + 1.0f);
    if (fabsf(s_neon - s_ref) > tol) {
        printf("FAIL q4k vecdot: neon=%g ref=%g (tol=%g)\n", s_neon, s_ref, tol); ++g_fails;
    }
}
```
Call `check_q4k_vecdot();` in `main` before the final pass/fail print.

- [ ] **Step 2: Build, confirm it fails to link**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -5
```
Expected: link error — `nnml_vec_dot_q4_K_q8_K` and `nnml_vec_dot_q4_K_q8_K_ref` naming not yet reconciled.

- [ ] **Step 3: Rename the scalar to `_ref`**

In `nnml/src/ops/types.cpp`, rename the function at line 437 from `nnml_vec_dot_q4_K_q8_K` to `nnml_vec_dot_q4_K_q8_K_ref` (body unchanged). In `nnml/include/ops.h`, update/add the declaration to match (`nnml_vec_dot_q4_K_q8_K_ref`). The trait at `types.cpp:60` still references `nnml_vec_dot_q4_K_q8_K` (the NEON name) — leave it.

- [ ] **Step 4: Add the x86 passthrough**

In `nnml/src/ops/x86/x86.cpp`, add (near the q6_K x86 impl at line 762):
```cpp
void nnml_vec_dot_q4_K_q8_K(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    // x86 SIMD Q4_K is out of scope; defer to the portable scalar reference.
    nnml_vec_dot_q4_K_q8_K_ref(n, s, bs, vx, bx, vy, by, nrc);
}
```
(Declare `nnml_vec_dot_q4_K_q8_K_ref` in `ops.h` so x86.cpp sees it.)

- [ ] **Step 5: Implement the NEON kernel + `_generic` fallback**

In `nnml/src/ops/arm/arm.cpp`, add (model on `nnml_vec_dot_q6_K_q8_K` at line 748):

```cpp
// scalar fallback for non-dotprod ARM (calls the portable _ref)
void nnml_vec_dot_q4_K_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    nnml_vec_dot_q4_K_q8_K_ref(n, s, bs, vx, bx, vy, by, nrc);
}

void nnml_vec_dot_q4_K_q8_K(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    NNML_UNUSED(nrc); NNML_UNUSED(bx); NNML_UNUSED(by); NNML_UNUSED(bs);

#if defined(__ARM_FEATURE_DOTPROD)
    const block_q4_K * NNML_RESTRICT x = (const block_q4_K *) vx;
    const block_q8_K * NNML_RESTRICT y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    float sum = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float d    = NNML_FP16_TO_FP32(x[i].d);
        const float dmin = NNML_FP16_TO_FP32(x[i].dmin);
        const uint8_t * qs     = x[i].qs;
        const int8_t  * q8     = y[i].qs;
        const float    yd      = y[i].d;

        // 8 sub-blocks of 32; pairs (lo/hi nibble) share a qs byte run.
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t sc, m; get_scale_min_k4(j, x[i].scales, &sc, &m);
            const float d_sub = d * sc;
            const float min_sub = dmin * m;
            const uint8_t * q32 = qs + j*32;     // 32 bytes -> 32 lo + 32 hi nibbles
            const int8_t  * y32 = q8  + j*32;

            // Unpack lo/hi nibbles to int8 vectors (0..15) and dot vs y32 via vdotq_s32.
            // (Implement the NEON nibble unpack + vdotq exactly as the q6_K kernel does
            //  its ql/qh unpack; accumulate sum_nq and sum_y per sub-block, then:
            //    sum += yd * (d_sub * sum_nq - min_sub * sum_y);
            //  Use the per-block dot math from the plan's "Q4_K dot math" section.)
            // ... NEON intrinsics (model on arm.cpp:748 q6_K's vdotq accumulation) ...
        }
        // (accumulate across blocks into `sum`; *s = sum at the end)
    }
    *s = sum;
#else
    nnml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}
```
**The implementer fills the NEON nibble-unpack + `vdotq_s32` accumulation** by reading `nnml_vec_dot_q6_K_q8_K` (`arm.cpp:748`) as the structural template and applying the Q4_K dot math (lo/hi nibbles → int8, dot vs `y_qs`, scale by `d*sc`, subtract `dmin*m*sum_y`). `get_scale_min_k4` and `quantize_row_q8_K` are already available. Declare `nnml_vec_dot_q4_K_q8_K` in `ops.h`.

- [ ] **Step 6: Build + run the test — PASS (NEON == scalar)**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -3
./build/nnml/test-tq2-0
```
Expected: `test-tq2-0: PASS` (q4k neon == ref within tolerance). If mismatch, the nibble-unpack or scale/min application is wrong — re-check against `dequantize_row_q4_K`.

- [ ] **Step 7: Full build clean**

```sh
cmake --build build -j 2>&1 | tail -10
```
Expected: success (no duplicate-symbol link errors — the rename + x86 passthrough reconcile the names).

- [ ] **Step 8: Commit**

```sh
git add nnml/src/ops/arm/arm.cpp nnml/src/ops/types.cpp nnml/include/ops.h nnml/src/ops/x86/x86.cpp nnml/test/test-tq2-0.cpp
git commit -m "Add NEON Q4_K vec_dot (dotprod); scalar becomes _ref fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Verify correctness + speedup on MiniCPM5-1B-Q4_K_M

**Files:** none — verification + a regression note.

- [ ] **Step 1: Correctness — greedy tokens match the pre-change scalar path**

Capture the MiniCPM5-1B output. Since the scalar path is now `_ref` (not directly selectable via a flag), the NEON path is the default; correctness is the unit test (NEON == _ref) + coherent end-to-end output:
```sh
./build/al-gen --model ./tmp/MiniCPM5-1B-Q4_K_M.gguf --prompt "The capital of France is" \
  --threads 8 --nodes 1 --numa none --max_gen 24 --print_model 0 --print_binding 0 --print_kv 0 2>&1 | grep -A2 "^> "
```
Expected: coherent continuation (e.g. mentions "Paris"). If garbage, the kernel has a bug (re-check the unit test and the nibble/scale math).

- [ ] **Step 2: Performance — big speedup over the 10.71 tok/s scalar baseline**

```sh
./build/al-gen --model ./tmp/MiniCPM5-1B-Q4_K_M.gguf --prompt "The capital of France is" \
  --threads 8 --nodes 1 --numa none --max_gen 48 2>&1 | grep "token/s"
```
Expected: decode tok/s substantially higher than 10.71 (target ~5–10× → ~50–90 tok/s). Record the number.

- [ ] **Step 3: Regression — the four bitcpm4 models (Q4_K embeddings) still correct**

```sh
for sz in 0.5b 1b; do
  ./build/al-gen --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-$sz-tq2_0.gguf \
    --prompt "The capital of France is" --threads 8 --nodes 1 --numa none --max_gen 12 2>&1 | grep -A1 "^> " | tr '\n' ' '; echo " ($sz)"
done
```
Expected: both still decode "Paris".

- [ ] **Step 4: Commit a verification note**

```sh
git commit --allow-empty -m "Verify NEON Q4_K: MiniCPM5-1B decode=<X> tok/s (was 10.71); bitcpm4 regress OK

MiniCPM5-1B-Q4_K_M: '<output>'  decode: <X> tok/s (scalar was 10.71)
bitcpm4 0.5b/1b: still 'Paris'

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(substitute the real numbers/outputs.)

---

## Self-Review (completed)

**Spec coverage:** §2 (NEON kernel modeled on q6_K) → Task 1 Step 5; §3 (wiring: rename→_ref, arm.cpp NEON, x86 passthrough) → Task 1 Steps 3–5; §4 (unit test NEON==_ref, end-to-end MiniCPM5, bitcpm4 regression, speedup) → Tasks 1 & 2; §5 risks (scale/nibble correctness → unit-test backstop; link wiring → Step 7 full build). ✓

**Placeholder scan:** the NEON nibble-unpack + `vdotq_s32` body is specified by reference to the local Q6_K template (`arm.cpp:748`) + the explicit Q4_K dot math, with the test as the correctness gate — same port-from-existing pattern used successfully for the TQ2_0 and Q4_0 kernels. No TBD.

**Type/name consistency:** `nnml_vec_dot_q4_K_q8_K` (NEON, arm.cpp + x86.cpp passthrough), `nnml_vec_dot_q4_K_q8_K_ref` (scalar, types.cpp), `nnml_vec_dot_q4_K_q8_K_generic` (arm.cpp fallback) — consistent across steps; the trait (`types.cpp:60`) unchanged.
