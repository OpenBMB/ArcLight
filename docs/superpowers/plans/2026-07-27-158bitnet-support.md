# 158BitNet (BitCPM / MiniCPM4 ternary) Model Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ArcLight run inference on the four `bitcpm4-{0.5b,1b,3b,8b}-tq2_0.gguf` ternary-weight models.

**Architecture:** Add a reference-scalar `NNML_TYPE_TQ2_0` quant type (dequantize + vec_dot) to the `nnml` backend, slotted into the existing generic matmul path exactly like Q4_K/Q6_K. The 1B/3B models (declared `general.architecture = llama`) then run unchanged through ArcLight's existing llama builder. The 0.5B/8B (declared `minicpm`) need a new `models/minicpm.cpp` registration that also applies MiniCPM depth scales (`embedding_scale`, `residual_scale`, `logit_scale`).

**Tech Stack:** C++17, CMake (out-of-tree), ArcLight `nnml` tensor backend + LLM layer. Reference: `/Users/chersu/workdir/AI/158BitNet` (C runtime whose scalar `bitnet_tq2_0_dequantize_block` we port).

## Global Constraints

- **C++17**, 4-space indent, braces on same line, `snake_case`, no formatter/linter — match surrounding style, avoid unrelated formatting churn.
- `NNML_TYPE_TQ2_0` **must equal 35** (the GGUF tensor type id) because `src/model.cpp:409` casts `ttype` directly to `nnml_type`. `NNML_TYPE_COUNT` must grow to cover index 35.
- This machine is **arm64 macOS** → build with `-DARCLIGHT_BACKEND=AUTO -DNNML_USE_NUMA=OFF`. The scalar TQ2_0 path is backend-agnostic (plain C++ in `types.cpp`).
- Model files live at `/Users/chersu/workdir/AI/158BitNet/models/` — **never commit them**; reference by absolute path.
- Branch is `add-158bitnet-support`. Commit after each task. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Reference-scalar only. No NEON/x86 TQ2_0 kernels, no LoRA, no server, no longrope/long-context work (out of scope).

---

## File Structure

| File | Responsibility | Task |
| --- | --- | --- |
| `nnml/include/ops.h` | `nnml_type` enum, op enum, `dequantize_row_*` / `vec_dot_*` declarations | 1 |
| `nnml/src/ops/types.cpp` | `type_traits[]` registry + all `dequantize_row_*` / `vec_dot_*` scalar impls | 1 |
| `nnml/test/test-tq2-0.cpp` | Unit test: TQ2_0 dequantize on hand-built blocks | 1 |
| `nnml/CMakeLists.txt` | Build `test-tq2-0` executable | 1 |
| `include/model.h` | `llm_hparams_item` enum | 3 |
| `nnml/include/cgraph.h` | `llm_hparams` struct fields (incl. `f_logit_scale`) | 3 |
| `src/model.cpp` | `load_gguf_kv` hparams switch | 3 |
| `models/minicpm.cpp` | New model: hparams/weights maps, rope config, ChatML template, scaled forward builder | 4 |

`nnml/src/ops/ops.cpp` and `models/llama.cpp` are **not modified** — TQ2_0 routes through the existing generic mul_mat default case, and the 1B/3B reuse the existing llama builder.

---

## Task 1: Add `NNML_TYPE_TQ2_0` quant type (dequantize + vec_dot)

**Files:**
- Modify: `nnml/include/ops.h` (enum at lines 141-172; decls near lines 340-354)
- Modify: `nnml/src/ops/types.cpp` (type_traits array at lines 35-77; new impls near q4_K impl ~line 428)
- Create: `nnml/test/test-tq2-0.cpp`
- Modify: `nnml/CMakeLists.txt` (after line 111)

**Interfaces:**
- Consumes: `block_tq2_0` (already defined in `nnml.h:730-733` as `{ uint8_t qs[QK_K/4]; nnml_half d; }`, qs-first — matches the ggml on-disk layout; do not reorder), `block_q8_0`, macros `QK_K` (=256), `NNML_FP16_TO_FP32`, `NNML_FP32_TO_FP16`, `NNML_RESTRICT`, `NNML_UNUSED`.
- Produces:
  - `void dequantize_row_tq2_0(const block_tq2_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);`
  - `void nnml_vec_dot_tq2_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc);`
  - `nnml_type NNML_TYPE_TQ2_0 = 35`; `NNML_TYPE_COUNT = 36`; `type_traits[35]` registered.

- [ ] **Step 1: Write the failing test**

Create `nnml/test/test-tq2-0.cpp`:

```cpp
#include <cstdio>
#include <cmath>
#include "ops.h"
#include "nnml.h"

static int g_fails = 0;

static void check(float got, float expv, const char * msg) {
    if (fabsf(got - expv) > 1e-5f) {
        printf("FAIL %s: got %g, expected %g\n", msg, got, expv);
        ++g_fails;
    }
}

int main(void) {
    // Block A: scale=1.0, all packed bytes zero -> every 2-bit code=0 -> output (0-1)*1.0 = -1.0
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0;
        float y[QK_K];
        dequantize_row_tq2_0(&b, y, QK_K);
        for (int i = 0; i < QK_K; ++i) check(y[i], -1.0f, "blockA all -1*d");
    }
    // Block B: scale=1.0, qs[0]=0xFF (all four 2-bit codes of byte 0 == 3), rest zero.
    // Dequant order within the first 32-byte group (j=0) emits 4 planes l=0..3 at output
    // offsets 0/32/64/96; for m=0 (qs[0]) every plane reads code 3 -> (3-1)*1.0 = 2.0.
    // m>=1 (qs[1..31]==0) -> -1.0. Second 32-byte group (j=32) all zero -> -1.0.
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(1.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0;
        b.qs[0] = 0xFF;
        float y[QK_K];
        dequantize_row_tq2_0(&b, y, QK_K);
        check(y[0],    2.0f, "blockB y[0]=2");
        check(y[32],   2.0f, "blockB y[32]=2");
        check(y[64],   2.0f, "blockB y[64]=2");
        check(y[96],   2.0f, "blockB y[96]=2");
        check(y[1],   -1.0f, "blockB y[1]=-1");
        check(y[128], -1.0f, "blockB y[128]=-1 (group 2)");
        check(y[255], -1.0f, "blockB y[255]=-1");
    }
    // Block C: scale=2.0, qs all 0xAA (low two bits = 0b10 = 2) -> (2-1)*2.0 = 2.0 everywhere
    {
        block_tq2_0 b;
        b.d = NNML_FP32_TO_FP16(2.0f);
        for (int i = 0; i < QK_K / 4; ++i) b.qs[i] = 0xAA;
        float y[QK_K];
        dequantize_row_tq2_0(&b, y, QK_K);
        for (int i = 0; i < QK_K; ++i) check(y[i], 2.0f, "blockC all (2-1)*2");
    }

    if (g_fails == 0) { printf("test-tq2-0: PASS\n"); return 0; }
    printf("test-tq2-0: FAIL (%d assertions)\n", g_fails);
    return 1;
}
```

- [ ] **Step 2: Register the test target so it compiles**

In `nnml/CMakeLists.txt`, after the `test-q4-k` block (lines 110-111), add:

```cmake

add_executable(test-tq2-0 ${CMAKE_CURRENT_SOURCE_DIR}/test/test-tq2-0.cpp)
target_link_libraries(test-tq2-0 PRIVATE nnml)
```

- [ ] **Step 3: Add the enum value**

In `nnml/include/ops.h`, replace the enum tail (lines 168-171):

```c
    NNML_TYPE_I32     = 26,
    NNML_TYPE_I64     = 27,
    NNML_TYPE_F64     = 28,
    NNML_TYPE_COUNT   = 29,
};
```

with:

```c
    NNML_TYPE_I32     = 26,
    NNML_TYPE_I64     = 27,
    NNML_TYPE_F64     = 28,
    // 29-34 reserved (unused GGUF type-id slots)
    NNML_TYPE_TQ2_0   = 35,
    NNML_TYPE_COUNT   = 36,
};
```

- [ ] **Step 4: Declare the dequantize + vec_dot functions**

In `nnml/include/ops.h`, after the `quantize_row_q8_K` declaration (line 354), add:

```c
void dequantize_row_tq2_0(const block_tq2_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);
void nnml_vec_dot_tq2_0_q8_0(int n, float * NNML_RESTRICT s, size_t bs, const void * NNML_RESTRICT vx, size_t bx, const void * NNML_RESTRICT vy, size_t by, int nrc);
```

- [ ] **Step 5: Run the test to verify it fails to link**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -5
```
Expected: link error — `dequantize_row_tq2_0` is declared but undefined (we have not implemented it yet). If the build is stale, reconfigure first: `cmake -S . -B build -DARCLIGHT_BACKEND=AUTO -DNNML_USE_NUMA=OFF`.

- [ ] **Step 6: Implement the dequantize + vec_dot**

In `nnml/src/ops/types.cpp`, just before the closing `// Q8_0` section's `dequantize_row_q8_0` (i.e. insert right after the `nnml_vec_dot_q4_K_q8_K` function that ends at line 457), add:

```cpp
// TQ2_0 — ternary {-1,0,+1}, 256 weights per super-block (QK_K), 64 packed bytes + one fp16 scale.
// Interleaved layout matching llama.cpp dequantize_row_tq2_0: each byte holds four 2-bit codes at
// bit-positions {0,2,4,6}; the loop emits (j-group, l-plane, m-element) order, 2*4*32 = 256 outputs.
void dequantize_row_tq2_0(const block_tq2_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = NNML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        for (int j = 0; j < QK_K / 4; j += 32) {   // 2 groups of 32 bytes (QK_K/4 == 64)
            for (int l = 0; l < 4; ++l) {          // 4 two-bit code planes per byte
                for (int m = 0; m < 32; ++m) {
                    const int8_t q = (qs[j + m] >> (l * 2)) & 3;  // code in {0,1,2,3}
                    *y++ = (float)(q - 1) * d;                     // {0,1,2} -> {-1,0,+1}; code 3 reserved
                }
            }
        }
    }
}

void nnml_vec_dot_tq2_0_q8_0(
        int n, float * NNML_RESTRICT s, size_t bs,
        const void * NNML_RESTRICT vx, size_t bx,
        const void * NNML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    NNML_UNUSED(bs);
    NNML_UNUSED(bx);
    NNML_UNUSED(by);

    const block_tq2_0 * NNML_RESTRICT x = (const block_tq2_0 *) vx;
    const block_q8_0  * NNML_RESTRICT y = (const block_q8_0  *) vy;
    const int nb = n / QK_K;
    float values[QK_K];
    float sum = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        dequantize_row_tq2_0(x + ib, values, QK_K);
        const float yd = NNML_FP16_TO_FP32(y[ib].d);
        for (int j = 0; j < QK_K; ++j) {
            sum += values[j] * yd * y[ib].qs[j];
        }
    }
    *s = sum;
}
```

- [ ] **Step 7: Register the type traits**

In `nnml/src/ops/types.cpp`, replace the array tail (lines 74-77):

```c
    make_type_traits("i32", 1, -1, sizeof(int32_t)),
    make_type_traits("i64", 1, -1, sizeof(int64_t)),
    make_type_traits("f64", 1, -1, sizeof(double)),
};
```

with:

```c
    make_type_traits("i32", 1, -1, sizeof(int32_t)),
    make_type_traits("i64", 1, -1, sizeof(int64_t)),
    make_type_traits("f64", 1, -1, sizeof(double)),
    place_holder,  // 29
    place_holder,  // 30
    place_holder,  // 31
    place_holder,  // 32
    place_holder,  // 33
    place_holder,  // 34
    make_type_traits("tq2_0", QK_K, -1, sizeof(block_tq2_0), true,
        (nnml_to_float_t) dequantize_row_tq2_0, nullptr, nullptr,
        nnml_vec_dot_tq2_0_q8_0, NNML_TYPE_Q8_0, 1),
};
```

- [ ] **Step 8: Build and run the test — verify PASS**

```sh
cmake --build build -j --target test-tq2-0 2>&1 | tail -5
./build/nnml/test-tq2-0
```
Expected: builds cleanly; output `test-tq2-0: PASS` (exit 0).

- [ ] **Step 9: Confirm the full library still builds (no regressions in the larger build)**

```sh
cmake --build build -j 2>&1 | tail -15
```
Expected: build succeeds (libnnml, libal, al-gen/chat/ppl, all tests). If a placeholder count mismatch appears (`type_traits[NNML_TYPE_COUNT]` initializer size), recheck Step 7 has exactly 36 entries (29 original + 6 `place_holder` + 1 tq2_0).

- [ ] **Step 10: Commit**

```sh
git add nnml/include/ops.h nnml/src/ops/types.cpp nnml/test/test-tq2-0.cpp nnml/CMakeLists.txt
git commit -m "Add NNML_TYPE_TQ2_0 ternary quant type (scalar dequantize + vec_dot)

Ports the reference scalar dequantize from 158BitNet so nnml can multiply
TQ2_0 ternary weight tensors via the existing generic matmul path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Verify the 1B and 3B (llama-arch) models run end-to-end

**Files:** none modified — pure verification that Task 1 unblocks the llama-arch models through ArcLight's existing `models/llama.cpp`.

**Interfaces:**
- Consumes: Task 1's `NNML_TYPE_TQ2_0`. The 1B/3B declare `general.architecture = llama` and store every projection as TQ2_0, token embeddings as Q4_K, output as Q6_K (all already-supported except TQ2_0, now added).

- [ ] **Step 1: Build the apps**

```sh
cmake --build build -j --target al-gen 2>&1 | tail -5
```
Expected: `build/al-gen` is produced.

- [ ] **Step 2: Run the 1B model with a short prompt**

```sh
./build/al-gen \
  --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf \
  --prompt "The capital of France is" \
  --threads 8 --nodes 1 --numa none --max_gen 16
```
Expected: model loads (prints logo + metadata), runs prefill + decode, prints generated text. No crash, no `NNML_ABORT`. The text should be plausibly coherent (e.g. mentions "Paris"). If you see `fatal error` / `abort` inside `forward_mul_mat` or a TQ2_0 assert, Task 1's vec_dot routing is wrong — recheck Step 7's `vec_dot_type = NNML_TYPE_Q8_0`.

- [ ] **Step 3: Run the 3B model the same way**

```sh
./build/al-gen \
  --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-3b-tq2_0.gguf \
  --prompt "The capital of France is" \
  --threads 8 --nodes 1 --numa none --max_gen 16
```
Expected: same — loads and generates without aborting.

- [ ] **Step 4 (strongest correctness oracle): cross-check the 1B against the 158BitNet reference runtime**

Build and run the reference runtime's minimal generator with the same model + prompt, then compare the first few generated tokens.

```sh
cmake -S /Users/chersu/workdir/AI/158BitNet -B /Users/chersu/workdir/AI/158BitNet/build >/dev/null 2>&1
cmake --build /Users/chersu/workdir/AI/158BitNet -j --target minimal_generate >/dev/null 2>&1
/Users/chersu/workdir/AI/158BitNet/build/minimal_generate \
  /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf "The capital of France is" 16
```
Expected: reference output that should closely match ArcLight's Step 2 output for the first several greedy tokens (tokenizers/templates may differ slightly at the edges; the first 4-8 tokens should agree). If they diverge immediately, suspect the TQ2_0 dequantize bit-interleaving (re-verify against the unit test) or the vec_dot accumulation. Record the two outputs in the commit message.

- [ ] **Step 5: Commit a verification note**

```sh
git commit --allow-empty -m "Verify bitcpm4 1B/3B (llama arch) inference on TQ2_0

ArcLight: '<recorded 1B output>'
158BitNet ref: '<recorded 1B output>'

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Plumb MiniCPM depth-scale hparams through the loader

The 0.5B and 8B carry `minicpm.embedding_scale` / `residual_scale` / `logit_scale`. ArcLight has `f_embedding_scale` / `f_residual_scale` fields (and `build_inp_embd` already applies `f_embedding_scale`) but no enum items, no loader cases, and no `f_logit_scale` field.

**Files:**
- Modify: `include/model.h` (enum `llm_hparams_item` at lines 29-60)
- Modify: `nnml/include/cgraph.h` (`llm_hparams` struct, fields at lines 218-220)
- Modify: `src/model.cpp` (`load_gguf_kv` switch, near lines 285-293)

**Interfaces:**
- Produces: enum items `LLM_EMBEDDING_SCALE`, `LLM_RESIDUAL_SCALE`, `LLM_LOGIT_SCALE`; struct field `llm_hparams::f_logit_scale`; loader cases that populate `hparams.f_embedding_scale` / `f_residual_scale` / `f_logit_scale`. Consumed by Task 4's `build_minicpm_forward`.

- [ ] **Step 1: Add the enum items**

In `include/model.h`, the `llm_hparams_item` enum ends (around line 59) with `LLM_QUANT_VERSION,` then `};`. Change it to:

```c
    LLM_QUANT_VERSION,
    LLM_EMBEDDING_SCALE,
    LLM_RESIDUAL_SCALE,
    LLM_LOGIT_SCALE,
};
```

- [ ] **Step 2: Add the `f_logit_scale` field**

In `nnml/include/cgraph.h`, the `llm_hparams` struct has (lines 218-220):

```c
    float    f_residual_scale       = 0.0;
    float    f_embedding_scale      = 0.0;
    float    f_attention_scale      = 0.0;
```

Immediately after `f_attention_scale`, add:

```c
    float    f_attention_scale      = 0.0;
    float    f_logit_scale          = 1.0;
```

- [ ] **Step 3: Add the loader cases**

In `src/model.cpp` `load_gguf_kv`, find the `case LLM_ATTN_VALUE_LENGTH:` block (lines 290-293):

```c
                    case LLM_ATTN_VALUE_LENGTH:
                        hparams.n_embd_head_v = r_u32(f);
                        LLM_LOG(is_print, "%u\n", hparams.n_embd_head_v);
                        break;
```

Immediately after that block's `break;`, insert:

```c
                    case LLM_EMBEDDING_SCALE:
                        hparams.f_embedding_scale = r_f32(f);
                        LLM_LOG(is_print, "%g\n", hparams.f_embedding_scale);
                        break;
                    case LLM_RESIDUAL_SCALE:
                        hparams.f_residual_scale = r_f32(f);
                        LLM_LOG(is_print, "%g\n", hparams.f_residual_scale);
                        break;
                    case LLM_LOGIT_SCALE:
                        hparams.f_logit_scale = r_f32(f);
                        LLM_LOG(is_print, "%g\n", hparams.f_logit_scale);
                        break;
```

- [ ] **Step 4: Build**

```sh
cmake --build build -j 2>&1 | tail -15
```
Expected: clean build. (No model uses these yet — Task 4 will.)

- [ ] **Step 5: Verify the scales load (using the existing llama path on the 0.5B is NOT possible — it's minicpm arch and has no builder yet. Instead, sanity-check by building and confirming no regressions on the llama models.)**

```sh
./build/al-gen \
  --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-1b-tq2_0.gguf \
  --prompt "hi" --threads 8 --nodes 1 --numa none --max_gen 4 2>&1 | tail -3
```
Expected: still runs (Task 2 behavior unchanged). Full verification of the scales happens in Task 4.

- [ ] **Step 6: Commit**

```sh
git add include/model.h nnml/include/cgraph.h src/model.cpp
git commit -m "Load MiniCPM depth scales (embedding/residual/logit_scale)

Adds hparam enum items, the f_logit_scale field, and loader cases. The
embedding scale is already applied by build_inp_embd; residual/logit are
applied in the upcoming minicpm forward builder.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add the `minicpm` model registration (0.5B + 8B)

**Files:**
- Create: `models/minicpm.cpp` (auto-globbed into `libal` by the root `CMakeLists.txt` `file(GLOB ... models/*.cpp)`)

**Interfaces:**
- Consumes: Task 1 (`NNML_TYPE_TQ2_0` for the projections); Task 3 (`hparams.f_embedding_scale` applied by `build_inp_embd`; `f_residual_scale` / `f_logit_scale` applied here). Existing `nnml_cgraph` builders (`build_inp_embd`, `build_norm`, `build_lora_mm`, `build_reshape_3d`, `build_rope_ext`, `build_attn`, `build_ffn`, `build_add`, `build_get_rows`), `nnml_scale(mem, NNML_TENSOR_TYPE_ACTIVATION, 0, 0, t, s)` + `graph.build_forward_expand(t)` for in-builder scaling (pattern from `cgraph.cpp:305-313`), and `graph.get_mem()`.
- Produces: a self-registering `"minicpm"` model (hparams map, weights map, rope config, ChatML template, `build_minicpm_forward` builder). The 0.5B and 8B route here automatically because their GGUF declares `general.architecture = minicpm`.

- [ ] **Step 1: Create `models/minicpm.cpp`**

Create the file with this content (mirrors `models/minicpm5.cpp` / `models/llama.cpp` structure; differs by `minicpm.*` key prefix, the three scale hparams, tied-output fallback, ChatML template, and the depth-scale application):

```cpp
// definition of minicpm (BitCPM / MiniCPM4 ternary, architecture key "minicpm")
#include "models.h"

std::map<std::string, llm_hparams_item> minicpm_hparams_map = {
    {"general.architecture",                        LLM_ARCHITECTURE},
    {"general.name",                                LLM_NAME},
    {"minicpm.block_count",                         LLM_N_LAYER},
    {"minicpm.context_length",                      LLM_CTX_LENGTH},
    {"minicpm.embedding_length",                    LLM_EMB_LENGTH},
    {"minicpm.feed_forward_length",                 LLM_FFN_LENGTH},
    {"minicpm.attention.head_count",                LLM_ATTN_HEADCOUNT},
    {"minicpm.attention.head_count_kv",             LLM_ATTN_HEADCOUNT_KV},
    {"minicpm.rope.freq_base",                      LLM_ROPE_FREQ_BASE},
    {"minicpm.attention.layer_norm_rms_epsilon",    LLM_ATTN_LNORM_EPS},
    {"minicpm.attention.key_length",                LLM_ATTN_KEY_LENGTH},
    {"minicpm.attention.value_length",              LLM_ATTN_VALUE_LENGTH},
    {"minicpm.embedding_scale",                     LLM_EMBEDDING_SCALE},
    {"minicpm.residual_scale",                      LLM_RESIDUAL_SCALE},
    {"minicpm.logit_scale",                         LLM_LOGIT_SCALE},
    {"tokenizer.ggml.model",                        LLM_TOKENIZER_MODEL},
    {"tokenizer.ggml.pre",                          LLM_TOKENIZER_PRE_MODEL},
    {"tokenizer.ggml.tokens",                       LLM_TOKENIZER_TOKENS},
    {"tokenizer.ggml.scores",                       LLM_TOKENIZER_SCORES},
    {"tokenizer.ggml.token_type",                   LLM_TOKENIZER_TOKEN_TYPE},
    {"tokenizer.ggml.merges",                       LLM_TOKENIZER_MERGES},
    {"tokenizer.ggml.eos_token_id",                 LLM_TOKENIZER_EOS_ID},
    {"tokenizer.ggml.padding_token_id",             LLM_TOKENIZER_PAD_ID},
    {"tokenizer.ggml.unknown_token_id",             LLM_TOKENIZER_UNK_ID},
    {"tokenizer.ggml.bos_token_id",                 LLM_TOKENIZER_BOS_ID},
    {"tokenizer.ggml.add_bos_token",                LLM_TOKENIZER_ADD_BOS},
    {"tokenizer.ggml.add_eos_token",                LLM_TOKENIZER_ADD_EOS},
    {"tokenizer.ggml.add_sep_token",                LLM_TOKENIZER_ADD_SEP},
    {"tokenizer.ggml.add_space_prefix",             LLM_TOKENIZER_ADD_SPA},
    {"tokenizer.chat_template",                     LLM_TOKENIZER_CHAT_TMPL},
    {"general.quantization_version",                LLM_QUANT_VERSION},
};
REGISTER_HPARAMS_MAP(minicpm, minicpm_hparams_map);

std::map<std::string, llm_weight_item> minicpm_weights_map = {
    {"token_embd.weight",        LLM_TOKEN_EMBEDDINGS},
    {"output.weight",            LLM_OUTPUT_PROJ},     // absent when tied (0.5B) — harmless
    {"output_norm.weight",       LLM_OUTPUT_NORM},
    {"attn_norm.weight",         LLM_ATTENTION_NORM},
    {"attn_q.weight",            LLM_ATTENTION_Q},
    {"attn_k.weight",            LLM_ATTENTION_K},
    {"attn_v.weight",            LLM_ATTENTION_V},
    {"attn_output.weight",       LLM_ATTENTION_O},
    {"ffn_down.weight",          LLM_FFN_DOWN},
    {"ffn_up.weight",            LLM_FFN_UP},
    {"ffn_gate.weight",          LLM_FFN_GATE},
    {"ffn_norm.weight",          LLM_FFN_NORM},
    {"rope_factors_long.weight", LLM_ROPE_LONG},
    {"rope_factors_short.weight",LLM_ROPE_SHORT},
};
REGISTER_WEIGHTS_MAP(minicpm, minicpm_weights_map);

REGISTER_ROPE_TYPE(minicpm, llm_rope_type::LLM_ROPE_TYPE_NORM);
REGISTER_FREQ_BASE(minicpm, 10000.0f);
REGISTER_FREQ_SCALE(minicpm, 1.0f);
REGISTER_EXT_FACTOR(minicpm, 0.0f);
REGISTER_ATTN_FACTOR(minicpm, 1.0f);
REGISTER_BETA_FAST(minicpm, 32.0f);
REGISTER_BETA_SLOW(minicpm, 1.0f);

std::string minicpm_apply_template(
    const std::vector<chat_msg> & messages,
    bool add_generation_prompt,
    bool enable_reasoning)
{
    std::string out;
    out.reserve(messages.size() * 128 + 512);
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & msg = messages[i];
        if (msg.role == "assistant") {
            append_block(out, "assistant", msg.content);
        } else if (msg.role == "user") {
            append_block(out, "user", msg.content);
        } else if (msg.role == "system") {
            append_block(out, "system", msg.content);
        }
    }
    if (add_generation_prompt) {
        out.append("<|im_start|>assistant\n");
        if (enable_reasoning) {
            out.append("<think>\n\n</think>\n\n");
        }
    }
    return out;
}
REGISTER_TEMPLATE_CALLER(minicpm, minicpm_apply_template);

void build_minicpm_forward(nnml_cgraph & graph, llm_model & model, bool is_tp) {
    if (is_tp) assert(graph.get_n_head_kv() % graph.get_n_para_graphs_max() == 0);
    int kv_parallel_size = is_tp ? graph.get_kvcache()->get_n_para_kvcaches() : 1;

    llm_hparams & hparams = graph.get_hparams();
    const int64_t n_embd_head = hparams.n_embd_head_v;
    NNML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    NNML_ASSERT(n_embd_head == hparams.n_rot);

    // embedding_scale is applied inside build_inp_embd (cgraph.cpp:58) when non-zero.
    nnml_tensor * cur;
    nnml_tensor * inpL = graph.build_inp_embd(model.tok_embd);
    nnml_tensor * inp_pos = graph.build_inp_pos(0);
    graph.build_attn_inp_kv();
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    nnml_tensor * inp_out_ids = graph.build_inp_out_ids(0);

    for (int il = 0; il < hparams.n_layer; ++il) {
        nnml_tensor * inpSA = inpL;
        cur = graph.build_norm(inpL, model.layers[il].tensors[ATTN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "attn_norm", il);
        {
            nnml_tensor_ptrs curs;
            if (is_tp) { curs = graph.build_scatter(cur, true); }
            else       { curs = cur; }

            nnml_tensor * rope_factors = model.get_rope_factors(il);
            nnml_tensor_ptrs Qcur = graph.build_lora_mm(model.layers[il].tensors[WQ], curs);
            nnml_tensor_ptrs Kcur = graph.build_lora_mm(model.layers[il].tensors[WK], curs);
            nnml_tensor_ptrs Vcur = graph.build_lora_mm(model.layers[il].tensors[WV], curs);
            Qcur = graph.build_reshape_3d(Qcur, n_embd_head, graph.get_n_head()/kv_parallel_size, graph.get_n_tokens());
            Kcur = graph.build_reshape_3d(Kcur, n_embd_head, graph.get_n_head_kv()/kv_parallel_size, graph.get_n_tokens());
            Vcur = graph.build_reshape_3d(Vcur, n_embd_head, graph.get_n_head_kv()/kv_parallel_size, graph.get_n_tokens());

            Qcur = graph.build_rope_ext(Qcur, inp_pos, rope_factors, graph.get_n_rot(), graph.get_rope_type(), graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());
            Kcur = graph.build_rope_ext(Kcur, inp_pos, rope_factors, graph.get_n_rot(), graph.get_rope_type(), graph.get_n_ctx_orig(), graph.get_freq_base(), graph.get_freq_scale(),
                    graph.get_ext_factor(), graph.get_attn_factor(), graph.get_beta_fast(), graph.get_beta_slow());

            curs = graph.build_attn(model.layers[il].tensors[WO], nullptr, Qcur, Kcur, Vcur,
                    nullptr, nullptr, nullptr, kq_scale, il);
            cur = is_tp ? graph.build_gather(curs, true) : curs;
        }
        // MiniCPM depth scale on the attention output (residual branch)
        if (hparams.f_residual_scale != 0.0f) {
            cur = nnml_scale(graph.get_mem(), NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_residual_scale);
            graph.build_forward_expand(cur);
        }
        if (il == hparams.n_layer - 1 && inp_out_ids && !graph.is_eval_mode()) {
            cur   = graph.build_get_rows(cur, inp_out_ids);
            inpSA = graph.build_get_rows(inpSA, inp_out_ids);
        }
        nnml_tensor * ffn_inp = graph.build_add(cur, inpSA);
        ffn_inp->set_name("%s-%d", "ffn_inp", il);

        cur = graph.build_norm(ffn_inp, model.layers[il].tensors[FFN_NORM][0], NULL, LLM_NORM_RMS, il);
        cur->set_name("%s-%d", "ffn_norm", il);

        nnml_tensor_ptrs curs;
        if (is_tp) { curs = graph.build_scatter(cur, false); }
        else       { curs = cur; }

        curs = graph.build_ffn(curs,
                        model.layers[il].tensors[FFN_UP],   NULL, NULL,
                        model.layers[il].tensors[FFN_GATE], NULL, NULL,
                        model.layers[il].tensors[FFN_DOWN], NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = is_tp ? graph.build_gather(curs, false) : curs;

        // MiniCPM depth scale on the FFN output (residual branch)
        if (hparams.f_residual_scale != 0.0f) {
            cur = nnml_scale(graph.get_mem(), NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_residual_scale);
            graph.build_forward_expand(cur);
        }

        cur = graph.build_add(cur, ffn_inp);
        inpL = cur;
    }
    cur = inpL;
    cur = graph.build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cur->set_name("%s-%d", "result_norm", -1);
    // tied output when output.weight is absent (0.5B)
    if (model.output == nullptr) model.output = model.tok_embd;
    cur = graph.build_lora_mm(model.output, cur);
    cur->set_name("%s-%d", "result_output", -1);
    // MiniCPM output logit scale
    if (hparams.f_logit_scale != 1.0f && hparams.f_logit_scale != 0.0f) {
        cur = nnml_scale(graph.get_mem(), NNML_TENSOR_TYPE_ACTIVATION, 0, 0, cur, hparams.f_logit_scale);
        graph.build_forward_expand(cur);
    }
    graph.set_t_logits(cur);
}

static bool _reg_minicpm_builder = (nnml_cgraph::reg_builder("minicpm", build_minicpm_forward), true);
```

- [ ] **Step 2: Build**

```sh
cmake --build build -j 2>&1 | tail -15
```
Expected: clean build (`models/minicpm.cpp` is auto-globbed into `libal`). If you see “graph builder not found for model minicpm” at runtime (not build time), the `_reg_minicpm_builder` static initializer isn't linked — confirm the root `CMakeLists.txt` globs `models/*.cpp` (it does).

- [ ] **Step 3: Verify the 0.5B runs**

```sh
./build/al-gen \
  --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-0.5b-tq2_0.gguf \
  --prompt "The capital of France is" \
  --threads 8 --nodes 1 --numa none --max_gen 16
```
Expected: loads (metadata shows `embedding_scale = 12`, `residual_scale = 0.2858`, `logit_scale = 4`), generates coherent text without abort. If output is garbage/nan, the most likely cause is a missing depth scale — confirm `--print_model 1` shows the three scales populated and that `build_inp_embd` scaled the embedding.

- [ ] **Step 4: Verify the 8B runs**

```sh
./build/al-gen \
  --model /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-8b-tq2_0.gguf \
  --prompt "The capital of France is" \
  --threads 8 --nodes 1 --numa none --max_gen 16
```
Expected: loads (8B: `embedding_scale = 12`, `residual_scale = 0.2475`, `logit_scale = 16`), generates. The 8B weight file is ~2.4 GB; the default `--w_gb 4` buffer covers it. If you hit a buffer-size abort, raise `--w_gb`.

- [ ] **Step 5 (correctness oracle): cross-check the 0.5B against the 158BitNet reference**

```sh
/Users/chersu/workdir/AI/158BitNet/build/minimal_generate \
  /Users/chersu/workdir/AI/158BitNet/models/bitcpm4-0.5b-tq2_0.gguf "The capital of France is" 16
```
Expected: first several greedy tokens agree with ArcLight's Step 3 output. Record both.

- [ ] **Step 6: Commit**

```sh
git add models/minicpm.cpp
git commit -m "Add minicpm model (BitCPM/MiniCPM4 ternary, 0.5B + 8B)

Registers the minicpm architecture with MiniCPM depth-scale handling
(embedding scale via build_inp_embd; residual + logit scales applied in
the forward builder) and tied-output fallback.

ArcLight 0.5B: '<recorded>'
158BitNet ref:  '<recorded>'

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed)

**Spec coverage:**
- Spec Component 1 (TQ2_0 type + dequantize + generic mul_mat) → Task 1. ✓ (No `ops.cpp` change needed — confirmed the default case at `ops.cpp:1277-1279` already routes non-Q4_0 to `forward_mul_mat_generic`, which uses the registered `vec_dot`.)
- Spec Component 2 (GGUF type-id mapping) → no change; confirmed `model.cpp:409` direct cast with `NNML_TYPE_TQ2_0 = 35`. ✓
- Spec Component 3 (1B/3B llama, no model change) → Task 2. ✓
- Spec Component 4 (0.5B/8B minicpm + scales) → Tasks 3 + 4. ✓
- Spec Component 5 (verification incl. 158BitNet cross-check) → Tasks 2.4, 4.5. ✓
- Spec risks 1–4 addressed: bit-interleave (unit test + reference cross-check), vec_dot requirement (mirrors q4_K), scale semantics (standard MiniCPM depth-scaling, applied via `nnml_scale`), longrope (out of scope, short-context verified). ✓

**Placeholder scan:** none — all code blocks are complete; all commands have expected output.

**Type consistency:** `dequantize_row_tq2_0` / `nnml_vec_dot_tq2_0_q8_0` names match across `ops.h` decl (Step 4), `types.cpp` impl (Step 6), and traits registration (Step 7). `LLM_EMBEDDING_SCALE`/`LLM_RESIDUAL_SCALE`/`LLM_LOGIT_SCALE` match across `model.h` (Task 3.1), `model.cpp` (Task 3.3), and `minicpm.cpp` hparams map (Task 4.1). `f_logit_scale` matches across `cgraph.h` (Task 3.2) and `minicpm.cpp` builder (Task 4.1).
