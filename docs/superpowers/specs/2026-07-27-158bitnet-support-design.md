# Design: Add 158BitNet (BitCPM / MiniCPM4 ternary) model support to ArcLight

- **Date:** 2026-07-27
- **Status:** Approved (pending spec review)
- **Branch:** `add-158bitnet-support`
- **Reference runtime:** `/Users/chersu/workdir/AI/158BitNet` (C inference runtime for OpenBMB BitCPM CANN GGUF models)

## 1. Goal

Make ArcLight able to load and run inference on the four 158BitNet model files:

```
bitcpm4-0.5b-tq2_0.gguf   arch=minicpm  24 layers  1024 hidden  scales(12.0 / 0.2858 / 4.0)   tied output
bitcpm4-1b-tq2_0.gguf     arch=llama    28 layers  2048 hidden  (no scales)                   Q6_K output, Q4_K embd
bitcpm4-3b-tq2_0.gguf     arch=llama    32 layers  2560 hidden  (no scales)                   Q6_K output, Q4_K embd
bitcpm4-8b-tq2_0.gguf     arch=minicpm  32 layers  4096 hidden  scales(12.0 / 0.2475 / 16.0)  MiniCPM4.1
```

**Architecture split:** `minicpm` = {0.5B, 8B} (MiniCPM4 / MiniCPM4.1, carry depth scales); `llama` = {1B, 3B} ("CPM 2B", plain llama). All four share the same transformer block and store **every projection as `TQ2_0` ternary** ({-1, 0, +1}, 2.0625 bits/weight). Token embeddings are Q4_K or F16; the output projection is Q6_K (1B/3B) or tied to the token embedding (0.5B; 8B to be confirmed at load — handled either way by the existing `if (model.output == nullptr) model.output = model.tok_embd;`). The "1.58-bit" in the name refers to the ternary weights, not the parameter count.

## 2. Key findings from exploration

- **ArcLight is ~90% there already.** The 1B/3B files declare `general.architecture = "llama"`, and ArcLight already ships a working `llama` builder + hparams/weights maps whose forward graph (GQA, RoPE, RMSNorm, SiLU FFN) is exactly what these models need. The 0.5B and 8B declare `"minicpm"` (MiniCPM4 / MiniCPM4.1) — same block, plus MiniCPM depth scales. `build_lora_mm` / `build_attn` / `build_ffn` all route through one op, `nnml_mul_mat(weight, activation)`.
- **The one essential missing piece is the `TQ2_0` matmul path.** ArcLight's `nnml_compute_forward_mul_mat` handles `Q4_0 / Q8_0 / Q4_K / Q6_K / F16 / F32 / I32` only. `NNML_TYPE_TQ2_0` is not in the type enum, so a TQ2_0 weight tensor cannot even be sized, let alone multiplied.
- **`block_tq2_0` is already defined** in `nnml/include/nnml.h` (`{ uint8_t qs[QK_K/4]; nnml_half d; }`, 66 bytes, qs-first) and is byte-identical in layout to the on-disk GGUF format and to 158BitNet's block (ArcLight reads GGUF bytes via raw `fread` into this struct).
- **The type-trait registry is index-driven and trivial to extend.** `type_traits[NNML_TYPE_COUNT]` (in `nnml/src/ops/types.cpp`) holds one entry per `nnml_type` enum value: `to_float` (dequantize), `vec_dot`, `blck_size`, `type_size`, etc. Q4_K/Q6_K work through the generic mul_mat path purely by registering traits — no mul_mat code changes.
- **GGUF tensor type → `nnml_type` is a direct cast** (`src/model.cpp:409`: `set_data_type((nnml_type)tensor_meta[i].ttype)`). Therefore `NNML_TYPE_TQ2_0` **must equal the GGUF tensor type id 35**, and `NNML_TYPE_COUNT` must grow to cover index 35 with a real entry (placeholder entries have `blck_size = -1` / `type_size = 0`, which would zero out the byte-size math at `model.cpp:415-525`).
- **`get_rope_factors` always returns `rope_short`** (`src/model.cpp:549`, comment: "just only use rope-short in the current version"). All four models declare `rope.scaling.type = longrope`; ArcLight does not implement position-dependent long/short switching. This is **correct for short-context verification** (within the 32K training length, longrope applies the short factors) and is a pre-existing ArcLight limitation, not new work.
- **MiniCPM depth scales are not loaded or applied anywhere.** `llm_hparams` has `f_embedding_scale` / `f_residual_scale` fields but no `llm_hparams_item` enum entries, no loaders, and the existing `minicpm5` builder ignores them. The 0.5B carries `embedding_scale = 12.0`, `residual_scale = 0.2857738`, `logit_scale = 4.0`; ignoring them gives incorrect output. The 1B/3B/8B (plain llama) have no such scales.
- 158BitNet provides both a **reference scalar** dequantize/dot (`bitnet_tq2_0_dequantize_block`, `bitnet_tq2_0_dot_product`) and heavily-optimized NEON/x86 kernels. **This design uses the scalar reference only.**

## 3. Decisions (from brainstorming)

- **Kernel depth:** reference scalar first. Optimize later.
- **Model scope:** all four models.
- **Kernel integration approach:** **A** — register `NNML_TYPE_TQ2_0` + port `dequantize_row_tq2_0`; reuse the existing generic mul_mat path (dequantize weight → f32, then f32 GEMM), exactly like Q4_K/Q6_K. Smallest, lowest-risk, correct. A dedicated int8 vec_dot (Approach B) is a clean drop-in optimization for later.

## 4. Components

### Component 1 — `nnml`: TQ2_0 type + reference dequantize  *(CORE; enables all 4 models)*

The only change required for 1B/3B/8B inference to work.

- **`nnml/include/ops.h`**
  - Add `NNML_TYPE_TQ2_0 = 35` to the `nnml_type` enum.
  - Bump `NNML_TYPE_COUNT` from `29` to `36` so `type_traits[NNML_TYPE_COUNT]` covers index 35.
  - Declare `void dequantize_row_tq2_0(const block_tq2_0 * NNML_RESTRICT x, float * NNML_RESTRICT y, int64_t k);` beside the other `dequantize_row_*` declarations.
- **`nnml/src/ops/types.cpp`**
  - Register `type_traits[35] = make_type_traits("tq2_0", QK_K, -1, sizeof(block_tq2_0), true, (nnml_to_float_t) dequantize_row_tq2_0, nullptr, nullptr, <vec_dot?>, <vec_dot_type>, 1)` — same shape as the Q4_K/Q6_K entries. Fill indices 27–34 with `place_holder` to preserve GGUF-id alignment (they already are placeholders conceptually; the array literal just needs enough entries).
  - Implement `dequantize_row_tq2_0` by porting 158BitNet's `bitnet_tq2_0_dequantize_block` from `158BitNet/src/quant_tq2_0.c`: for each 66-byte block, read fp16 scale `d`, then unpack the 64 packed bytes into 256 ternary values using the llama.cpp/ggml 2-bit interleave (4 codes per byte, interleaved by 32-element strides), map codes → `{-1, 0, +1}`, and multiply by `d`. Adapt field access to ArcLight's `d`-first struct.
- **`nnml/src/ops/ops.cpp`**
  - Confirm `nnml_compute_forward_mul_mat` routes `NNML_TYPE_TQ2_0` to the generic path (the default case at `ops.cpp:1278` already calls `forward_mul_mat_generic`, which uses the registered `to_float`). Add an explicit `case NNML_TYPE_TQ2_0:` falling through to generic if the switch structure requires it.
  - If the generic path requires a `vec_dot` (rather than `to_float` alone — to confirm in planning), add a trivial `nnml_vec_dot_tq2_0_f32` (dequantize the row, then f32 dot against the activation row), analogous to `nnml_vec_dot_q4_K_q8_K`, and point the trait's `vec_dot` / `vec_dot_type` at it.
- **No changes** to `nnml/src/ops/{arm,x86}/` — the reference kernel is plain C++ living in `types.cpp`.

### Component 2 — GGUF tensor-type mapping: **no change**

The cast `(nnml_type)ttype` already handles id 35 once the enum + traits exist (Component 1). Confirmed at `src/model.cpp:409,415-525`.

### Component 3 — 1B/3B (llama arch): **no model-code change**

These reuse ArcLight's existing `llama` builder + maps unchanged. Token embeddings (Q4_K) and output (Q6_K) are already-supported types; all projections (TQ2_0) are covered by Component 1. Longrope uses `rope_short` (correct within 32K context).

### Component 4 — 0.5B and 8B (minicpm arch): new registration + MiniCPM depth scales

Both the 0.5B (MiniCPM4) and 8B (MiniCPM4.1) declare `general.architecture = "minicpm"` with `minicpm.*` keys and MiniCPM depth scales ArcLight currently ignores (the per-model values differ — e.g. 0.5B logit_scale=4.0, 8B logit_scale=16.0 — so they must be **loaded from the GGUF, not hardcoded**). The 0.5B has tied output; the 8B's output path is handled either way by the existing tied-output fallback.

- **`include/model.h`**
  - Add `LLM_EMBEDDING_SCALE`, `LLM_RESIDUAL_SCALE`, `LLM_LOGIT_SCALE` to the `llm_hparams_item` enum.
  - Add `float f_logit_scale = 1.0f;` to `llm_hparams` (`f_embedding_scale` and `f_residual_scale` already exist).
- **`src/model.cpp` (`load_gguf_kv`)**
  - Add cases mapping `minicpm.embedding_scale` → `LLM_EMBEDDING_SCALE` (`f_embedding_scale`), `minicpm.residual_scale` → `LLM_RESIDUAL_SCALE` (`f_residual_scale`), `minicpm.logit_scale` → `LLM_LOGIT_SCALE` (`f_logit_scale`).
- **New `models/minicpm.cpp`** (mirror the structure of `models/minicpm5.cpp` / `models/llama.cpp`):
  - `minicpm_hparams_map`: `minicpm.block_count`, `minicpm.context_length`, `minicpm.embedding_length`, `minicpm.feed_forward_length`, `minicpm.attention.head_count`, `minicpm.attention.head_count_kv`, `minicpm.attention.layer_norm_rms_epsilon`, `minicpm.rope.freq_base`, `minicpm.attention.key_length`, `minicpm.attention.value_length`, the three `*_scale` keys, tokenizer keys, `tokenizer.chat_template`, `general.quantization_version`.
  - `minicpm_weights_map`: `token_embd.weight`, `output_norm.weight`, `attn_norm.weight`, `attn_q/k/v/output.weight`, `ffn_norm/gate/up/down.weight`, `rope_factors_long.weight`, `rope_factors_short.weight`. **No** `output.weight` (tied).
  - `REGISTER_ROPE_TYPE(minicpm, LLM_ROPE_TYPE_NORM)`, `REGISTER_FREQ_BASE`, `REGISTER_FREQ_SCALE`, `REGISTER_EXT_FACTOR(0)`, `REGISTER_ATTN_FACTOR(1)`, `REGISTER_BETA_FAST/SLOW`.
  - `minicpm_apply_template`: ChatML (`<|im_start|>role\n…<|im_end|>\n`) matching the GGUF `chat_template`.
  - `build_minicpm_forward`: structurally the MiniCPM/lllama block, but applying the depth scales — `embedding_scale` on the `build_inp_embd` output, `residual_scale` on both residual adds (attn-out add and ffn-out add), `logit_scale` on the output projection; tied output (`if (model.output == nullptr) model.output = model.tok_embd;`).
  - `static bool _reg_minicpm_builder = (nnml_cgraph::reg_builder("minicpm", build_minicpm_forward), true);`
- **Scale application mechanism (to finalize in planning):** confirm what scalar-multiply / scaled-add graph ops already exist in `nnml_cgraph`. Prefer composing from an existing op; otherwise add minimal scaled helpers. This is the one design point not yet nailed down.

### Component 5 — Verification

- Configure + build: `cmake -S . -B build -DARCLIGHT_BACKEND=AUTO && cmake --build build -j`. (This Mac is arm64 → NEON backend; the scalar TQ2_0 path is backend-agnostic and lives in `types.cpp`.)
- Models live at `/Users/chersu/workdir/AI/158BitNet/models/` — **must not be committed**; reference by absolute path.
- For each model: `./build/al-gen --model …/bitcpm4-<size>-tq2_0.gguf --prompt "The capital of France is"`; confirm coherent text and a non-crashing decode loop. Use `--numa none --nodes 1 --threads <N>` (single-node) to keep the first verification simple.
- **Strongest correctness oracle:** 158BitNet is buildable on this machine. Cross-check first-token logits and short greedy generations against `158BitNet/build/minimal_generate …/bitcpm4-0.5b-tq2_0.gguf "…"` and the 1B for the same prompt.
- A focused `nnml/test/test-tq2-0.cpp` that round-trips a known vector through `dequantize_row_tq2_0` and checks against 158BitNet's scalar reference is a cheap, fast unit guardrail (register it in `nnml/CMakeLists.txt`).

## 5. Risks / things to confirm during planning

1. **TQ2_0 bit-interleaving** must match the on-disk layout byte-for-byte. Validate against 158BitNet's scalar `bitnet_tq2_0_dequantize_block`. Both ArcLight's and 158BitNet's structs are `qs`-first (`{ uint8_t qs[QK_K/4]; nnml_half d; }`), matching the ggml on-disk TQ2_0 layout; the dequant accesses `.d`/`.qs` by name so it ports directly — do **not** reorder the struct fields (a `d`-first struct would misread the raw GGUF bytes and silently break all four models).
2. **Generic mul_mat** may require a registered `vec_dot` rather than `to_float` alone. If so, add the trivial f32 vec_dot — low risk.
3. **MiniCPM scale semantics** — confirm embedding_scale multiplies token embeddings, residual_scale multiplies the attention/FFN outputs added into the residual stream, and logit_scale multiplies the final logits (standard MiniCPM depth-scaling). Confirm the existing graph op set for scalar multiply / scaled add.
4. **longrope** uses `rope_short` only (pre-existing ArcLight behavior) — correct for short-context verification; long-context accuracy is an existing limitation, explicitly out of scope.

## 6. Explicitly out of scope

Optimized NEON/x86 TQ2_0 kernels (Approaches B/C), LoRA loading, the OpenAI-compatible HTTP server, Q8 KV-cache mode, pipeline parallelism (`--numa pp`), cross-NUMA tensor-parallel verification of ternary models, and long-context longrope fixes. These are natural follow-ups.

## 7. File touch-summary

| File | Change |
| --- | --- |
| `nnml/include/ops.h` | Add `NNML_TYPE_TQ2_0 = 35`; bump `NNML_TYPE_COUNT` to 36; declare `dequantize_row_tq2_0` |
| `nnml/src/ops/types.cpp` | Register `type_traits[35]`; implement `dequantize_row_tq2_0` (+ optional `nnml_vec_dot_tq2_0_f32`) |
| `nnml/src/ops/ops.cpp` | Route `NNML_TYPE_TQ2_0` to generic mul_mat (likely already the default) |
| `include/model.h` | Add `LLM_EMBEDDING_SCALE`/`LLM_RESIDUAL_SCALE`/`LLM_LOGIT_SCALE` enum items; `f_logit_scale` field |
| `src/model.cpp` | Load the three `minicpm.*_scale` keys in `load_gguf_kv` |
| `models/minicpm.cpp` | **New file** — hparams/weights maps, rope config, ChatML template, `build_minicpm_forward` (applies scales, tied output), builder registration |
| `nnml/test/test-tq2-0.cpp` | **New file** (optional guardrail) — dequantize round-trip vs 158BitNet reference; register in `nnml/CMakeLists.txt` |

1B/3B require **no** model-file changes — they reuse the existing `models/llama.cpp`. 0.5B and 8B share the new `models/minicpm.cpp`.
