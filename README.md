# ✨ArcLight

![logo](https://github.com/user-attachments/assets/5249801e-02ea-4c10-ba81-2d36e8b26e87)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/OpenBMB/ArcLight)](https://github.com/OpenBMB/ArcLight/releases)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)

ArcLight: A Lightweight LLM Inference Framework

## 🤩 What's this?

![logo](https://github.com/user-attachments/assets/2b836ebb-ec57-41bc-aa3e-059efe983292)


**ArcLight** is designed to building a **lightweight**, **easily optimized**, unified memory-oriented LLM inference framework in **C/C++**. **Unified memory** refers to the possible heterogeneous computing units sharing the same physical main memory. In other words, ArcLight will be adaptable to all inference scenarios beyond high-performance GPU servers in the future.

In the v1.0 release we just published, we introduced targeted optimizations for **many-core CPU platforms**. A many-core platform refers to a machine equipped with dozens of CPU cores, which are typically organized and managed across multiple *NUMA nodes*. On such systems, cross-node memory access often becomes a severe performance bottleneck and significantly limits inference efficiency. To address this issue, we introduce **cross-node tensor parallelism on CPU** cores for the first time, enabling substantial acceleration of inference. Compared with the widely used llama.cpp, our system achieves up to 50% higher inference throughput under the same core count, and on some machines the speedup can reach as high as 100% (2×). Please refer to our paper for details (preprint soon).

----

## 🚀 Quick start

Currently, ArcLight provides operator adaptations only for the ARM platform; support for x86 and other architectures will be available soon. At present, the only way to use ArcLight is to *build from source* by cloning this repository. As the project continues, we will also provide precompiled installation packages and Docker images.

We sincerely invite developers from around the world to participate in the project and help build a high-performance unified-memory LLM inference framework together.

Before running ArcLight, you need to [download](https://huggingface.co) a model. ArcLight uses the GGUF model format from the [llama.cpp](https://github.com/ggml-org/llama.cpp) project. Instructions for converting models to the GGUF format can be found in the llama.cpp documentation. Currently, we have only defined support for the Qwen3 model. We recommend that users start by trying Qwen3-4B. Contributions are welcome—developers are encouraged to define additional models and contribute them to this project.

Example command:

```sh
# Use a local model to generate
al-gen --model /home/xyz/Qwen3-4B-Q4_0.gguf --prompt "Hello!"

# Or chat with model
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

Make sure that your machine has the related toolkit, e.g., GCC/G++ (at least supporting C++ 17).


## 💻 Command-line Arguments

Currently, we provide the essential command-line arguments required to run inference. Please use them according to the descriptions below.

- `--model`: model path, required
- `--prompt`: prompt, 'Hello!' is default
- `--threads`: number of threads using in the inference
- `--nodes`: number of numa nodes when using tp/pp paralism
- `--max_length`: maximum context length, you should not use the longer context
- `--max_gen`: maximum generation length in one round
- `--fattn`: switch on flash attention, can not be 0 (false) in this version
- `--asm`: switch on use asm operator, should be 1 (true) and we will implement to automatically detect
- `--print_model`: whether to print model metadata when loading one model
- `--print_binding`: whether to print the thread-core binding when launch the app
- `--print_kv`: whether to print the overhead of KV cache
- `--print_perf`: whether to print the profile (time/speed) when exit app

We support to inference mode now, i.e. single node mode and multi-node mode. They can be use as follows:

- `--numa none`: single node mode. We prioritize using all cores on the NUMA node where the program is launched. However, if the number of threads specified exceeds the cores available on a single node, cores from other nodes *will also be used*, which may **impact performance**.
- `--numa tp`: multi-node tensor parallelism. Use cross-node tensor parallelism. Tensors and threads are evenly distributed across NUMA nodes to achieve **maximum performance**. Please note that the number of nodes to use must be set with `--nodes N`, and currently it must be a power of 2.

We also pretend to support pipeline parallelism in the fure. Hence the argument `--numa pp` can be use soon. 

The current v1.0 release requires manually setting the sizes of various buffers, including weight, activation, KV cache, and thread group workspace, with the unit in GB. For example, one usage is `--w_gb 4 --a_gb 8 --kv_gb 2 --work_gb 2`. We will soon integrate functionality to automatically detect and configure these buffers.

## 📝 TODO list

ArcLight is a project that we aim to maintain and improve over the long term. We are fully aware that the framework still has many imperfections and missing features. Once again, we sincerely invite open-source contributors from around the world to participate in the development of this meaningful framework. Below, we have listed the areas/topics that require further improvement in the future:

- 🔥 Add x86-64 arch support
- 🔥 Add non-standard attention implementation
- Add other model series (Llama/MiniCPM etc.)
- Optimize the Scatter/Gather operators
- Add cross-NUMA pipeline parallelism
- Compile and test on Windows
- Add GPU support on Edge-device
- Organize modules to make their boundaries clear
- Refactor the code to improve readability
- Refactor the KV cache management
- Optimize the hardware-related operators
- Create and edit documents

----

## 🤗 Acknowledgement

We draw lots of design inspiration from the popular framework [llama.cpp](https://github.com/ggml-org/llama.cpp). In addition, our v1.0 release almost fully transplants its operator library and KV cache management approach. We would like to express our heartfelt thanks to the initiators and contributors of that project!
