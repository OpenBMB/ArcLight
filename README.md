# ✨ ArcLight

![logo](https://github.com/user-attachments/assets/5249801e-02ea-4c10-ba81-2d36e8b26e87)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/OpenBMB/ArcLight)](https://github.com/OpenBMB/ArcLight/releases)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)

ArcLight: A Lightweight LLM Inference Framework

## 🤩 What's this?

![logo](https://github.com/user-attachments/assets/2b836ebb-ec57-41bc-aa3e-059efe983292)


**ArcLight** is designed to provide a **lightweight**, **easy-to-optimize**, unified-memory-oriented LLM inference framework in **C/C++**. **Unified memory** refers to systems where heterogeneous computing units can share the same physical main memory. In other words, ArcLight is designed for inference scenarios beyond high-performance GPU servers.

In the v1.0 release we just published, we introduced targeted optimizations for **many-core CPU platforms**. A many-core platform refers to a machine equipped with dozens of CPU cores, which are typically organized and managed across multiple *NUMA nodes*. On such systems, cross-node memory access often becomes a severe performance bottleneck and significantly limits inference efficiency. To address this issue, we introduce **cross-node tensor parallelism on CPU** cores for the first time, enabling substantial acceleration of inference. Compared with the widely used llama.cpp, our system achieves up to 50% higher inference throughput under the same core count, and on some machines the speedup can reach as high as 100% (2×). Please refer to our paper for details (preprint soon).

----

## 🚀 Quick start

ArcLight currently provides CPU backends for both ARM and x86 platforms, together with basic Windows build support. At present, the recommended way to use ArcLight is to *build from source* by cloning this repository. As the project evolves, we will also provide precompiled packages and Docker images.

We sincerely invite developers from around the world to participate in the project and help build a high-performance unified-memory LLM inference framework together.

Before running ArcLight, you need to [download](https://huggingface.co) a model. ArcLight uses the GGUF model format from the [llama.cpp](https://github.com/ggml-org/llama.cpp) project. Instructions for converting models to GGUF can be found in the llama.cpp documentation. The current codebase includes model definitions for Qwen3, Llama, MiniCPM5, and BitCPM/MiniCPM4 (1.58-bit ternary, `TQ2_0`; the `bitcpm4-*` models). We recommend starting with Qwen3-4B or another small GGUF model for initial testing. Contributions for additional model families are welcome.

Example command:

```sh
# Use a local model to generate
al-gen --model /home/xyz/Qwen3-4B-Q4_0.gguf --prompt "Hello!"

# Or chat with the model
al-chat --model /home/xyz/Qwen3-4B-Q4_0.gguf

# Or evaluate perplexity on one text
al-ppl --model /home/xyz/Qwen3-4B-Q4_0.gguf --prompt "Good morning, Miss Lee!"
```

Build from source code:
```sh
git clone https://github.com/OpenBMB/ArcLight.git
cd ArcLight

mkdir build && cd build
cmake ..
make -j 32
```

You can also specify CMake options explicitly. For example, on Linux:

```sh
cmake -B build -DARCLIGHT_BACKEND=AUTO -DNNML_USE_NUMA=ON
cmake --build build --config Release -j 32
```

`ARCLIGHT_BACKEND` can be set to `AUTO`, `NEON`, `X86`, or `NONE`. In most cases, `AUTO` is recommended because it selects the backend according to the target CPU architecture.

On Windows with Visual Studio, you can build with:

```bat
cmake -B build -G "Visual Studio 18 2026"
cmake --build build --config Release -j 32
```

Make sure that your machine has the required toolchain, such as GCC/G++ or MSVC, with C++17 support.


## 💻 Command-line Arguments

Currently, we provide the essential command-line arguments required to run inference. Please use them according to the descriptions below.

- `--model`: model path, required
- `--prompt`: prompt; the default is `Hello!`
- `--threads`: number of threads used for inference
- `--nodes`: number of NUMA nodes used for TP/PP parallelism
- `--max_length`: maximum context length; the input context should not exceed this value
- `--max_gen`: maximum number of tokens generated in one round
- `--fattn`: enable flash attention; this cannot be `0` (false) in the current version
- `--asm`: enable assembly-optimized operators; this should be `1` (true) for now, and automatic detection will be added later
- `--print_model`: whether to print model metadata when loading one model
- `--print_binding`: whether to print the thread-core binding when launching the app
- `--print_kv`: whether to print the overhead of KV cache
- `--print_perf`: whether to print the performance profile (time/speed) when exiting the app

We currently support two inference modes: single-node mode and multi-node mode. They can be used as follows:

- `--numa none`: single node mode. We prioritize using all cores on the NUMA node where the program is launched. However, if the number of threads specified exceeds the cores available on a single node, cores from other nodes *will also be used*, which may **impact performance**.
- `--numa tp`: multi-node tensor parallelism. Use cross-node tensor parallelism. Tensors and threads are evenly distributed across NUMA nodes to achieve **maximum performance**. Please note that the number of nodes to use must be set with `--nodes N`, and currently it must be a power of 2.

We also plan to support pipeline parallelism in the future. The `--numa pp` mode will be enabled once that implementation is ready.

The current v1.0 release requires manually setting the sizes of several buffers, including the weight buffer, activation buffer, KV cache, and thread-group workspace. These values are specified in GB. For example: `--w_gb 4 --a_gb 8 --kv_gb 2 --work_gb 2`. We will integrate automatic buffer sizing in a future release.

## 📊 Performance

On ARM, every quantized matmul path is now NEON-accelerated (`TQ2_0`/`Q4_0`/`Q4_K`/`Q6_K`/`Q8_0`/`F16`): the ternary `TQ2_0` path uses a load-time I2S reorder + `vdotq_s32` kernel (ported from the [158BitNet](https://github.com/OpenBMB/158BitNet) runtime), and `Q4_K` uses a NEON dotprod `vec_dot`. The decode (token-generation) throughput below is measured single-node on Apple Silicon (M4, 4 P-cores + 6 E-cores, `--threads 8 --nodes 1 --numa none`).

**1.58-bit ternary models** (`bitcpm4-*`, `TQ2_0` projections, I2S NEON kernel):

| Model | decode (tok/s) |
| --- | ---: |
| `bitcpm4-0.5b-tq2_0.gguf` | ~188 |
| `bitcpm4-1b-tq2_0.gguf`   | ~112 |
| `bitcpm4-3b-tq2_0.gguf`   | ~53  |
| `bitcpm4-8b-tq2_0.gguf`   | ~21  |

**Standard-quant model** (`Q4_K` projections, NEON `vec_dot`):

| Model | decode (tok/s) |
| --- | ---: |
| `MiniCPM5-1B-Q4_K_M.gguf` | ~76 |

For reference, these were ~9–10× (`bitcpm4-0.5b`: 19.6 → 188 tok/s) and ~7× (`MiniCPM5-1B-Q4_K_M`: 10.7 → 76 tok/s) faster than the scalar fallback before the NEON kernels. Decode throughput is sensitive to thread count: on this M4, `--threads 8` outperformed both fewer threads and the full 10 (the efficiency cores add contention past 8 on this bandwidth/compute-mixed workload).

## 📝 TODO list

ArcLight is a project that we aim to maintain and improve over the long term. We are fully aware that the framework still has many imperfections and missing features. Once again, we sincerely invite open-source contributors from around the world to participate in the development of this meaningful framework. Below, we have listed the areas/topics that require further improvement in the future:

- 🔥 Add non-standard attention implementations
- Add more model series and improve model coverage
- Optimize the Scatter/Gather operators
- Add cross-NUMA pipeline parallelism
- Add GPU support on edge devices
- Organize modules to make their boundaries clear
- Refactor the code to improve readability
- Refactor the KV cache management
- Optimize the hardware-related operators
- Improve documentation

----

## 📝 Citation

If you use ArcLight in your research, please cite our work:

```bibtex
@inproceedings{xu2026arclight,
  title={ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs},
  author={Xu, Yuzhuang and Han, Xu and Li, Yuxuan and Che, Wanxiang},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics: System Demonstration},
  year={2026},
  url={https://arxiv.org/pdf/2603.07770}
}
```

## 🤗 Acknowledgement

We draw lots of design inspiration from the popular framework [llama.cpp](https://github.com/ggml-org/llama.cpp). In addition, our v1.0 release almost fully transplants its operator library and KV cache management approach. We would like to express our heartfelt thanks to the initiators and contributors of that project!
