#include <stdio.h>
#include <signal.h>
#include <atomic>
#include <iostream>

#include "arg.h"
#include "model.h"
#include "models.h"
#include "context.h"
#include "ops.h"
#include "scheduler.h"


arg_parser args;

std::atomic<bool> g_interrupted(false);
std::atomic<bool> g_exit(false);

decoding_context * g_ctx = nullptr;

void sigint_handler(int) {
    // if g_exit is true, it means we need to interrupt the inference;
    // if g_exit is false, we can just exit directly
    if (g_exit) {
        g_interrupted = true;
    } else {
        printf("\n");
        auto exit_callback = [&](decoding_context * ctx) {
            ctx->print_perf();
        };
        exit_callback(g_ctx);
        exit(0);
    }
}

int32_t main(int32_t argc, char** argv) {
    print_logo();
    signal(SIGINT, sigint_handler);

    init_args();
    args.parse(argc, argv);
    bool    is_asm = args.get<bool>("asm");
    bool    is_fattn = args.get<bool>("fattn");
    bool    is_print_model = args.get<bool>("print_model");
    bool    is_print_binding = args.get<bool>("print_binding");
    bool    is_print_kv = args.get<bool>("print_kv");
    bool    is_print_perf = args.get<bool>("print_perf");
    int32_t n_threads = args.get<int32_t>("threads");
    int32_t n_nodes = args.get<int32_t>("nodes");
    int32_t w_gb = args.get<int32_t>("w_gb");
    int32_t a_gb = args.get<int32_t>("a_gb");
    int32_t kv_gb = args.get<int32_t>("kv_gb");
    int32_t work_gb = args.get<int32_t>("work_gb");
    int32_t max_length = args.get<int32_t>("max_length");
    int32_t max_gen = args.get<int32_t>("max_gen");
    std::string model_path = args.get<std::string>("model");
    std::string numa_strategy = args.get<std::string>("numa");
    std::string prompt = args.get<std::string>("prompt");
    nnml_type type_v = args.get_enum<nnml_type>("tk");
    nnml_type type_k = args.get_enum<nnml_type>("tv");
    if (numa_strategy == "none" && n_nodes != 1) abort();
    if ((numa_strategy == "tp" || numa_strategy == "pp") && n_nodes <= 1) abort();  // here we only chech if n_nodes > 1; but in fact n_nodes should be power of 2 in this version

    int32_t n_thread_per_node;
    if (numa_strategy == "none") {
        n_thread_per_node = n_threads;
    } else if (numa_strategy == "tp") {
        n_thread_per_node = n_threads / n_nodes;
    } else if (numa_strategy == "pp") {
        throw std::runtime_error("not implemented yet");
        // n_thread_per_node = n_threads / n_nodes;
    } else {
        throw std::runtime_error("unknown numa strategy: " + numa_strategy);
    }
    bool is_tp = (numa_strategy == "tp") ? true : false;

    // load model and hparams
    llm_model model;
    model.is_asm_gemm = is_asm;
    model.n_tensor_nodes = n_nodes;
    model.open_model_file(model_path.c_str());
    model.load_gguf_kv(is_print_model);
    model.hparams.rope_type = get_model_rope_type(model.name);
    model.hparams.f_freq_base = get_model_freq_base(model.name);
    model.hparams.f_freq_scale = get_model_freq_scale(model.name);
    model.hparams.f_ext_factor = get_model_ext_factor(model.name);
    model.hparams.f_attn_factor = get_model_attn_factor(model.name);
    model.hparams.f_beta_fast = get_model_beta_fast(model.name);
    model.hparams.f_beta_slow = get_model_beta_slow(model.name);
    
    // allocate memory
    nnml_memory::cparams params;
    nnml_affinity_mode affinity_mode;
    if (numa_strategy == "none") {
        params.strategy = NNML_ALLOC_STRATEGY_UMA;
        affinity_mode = NUMA_UNIFIED;
    } else if (n_nodes > 1) {
        params.strategy = NNML_ALLOC_STRATEGY_NUMA;
        affinity_mode = NUMA_DISTRIBUTED;
    } else {
        throw std::runtime_error("invalid numa strategy: " + numa_strategy);
    }
    params.n_nodes_manual = n_nodes;
    params.total_wsize_bytes = (std::size_t)w_gb GB;
    params.total_csize_bytes = (std::size_t)a_gb GB;
    params.total_kvsize_bytes = (std::size_t)kv_gb GB;
    params.tmp_work_size_bytes = (std::size_t)work_gb GB;
    params.zero_init = true;
    auto mem = nnml_memory::create(params);
    uint8_t ** temp_work_data = new uint8_t * [n_nodes];
    for (size_t i = 0; i < params.n_nodes_manual; ++i) {
        temp_work_data[i] = (uint8_t *) mem->tmp_work_buffer(i)->base();
    }
    size_t temp_work_size = mem->tmp_work_buffer(0)->size();
    model.load_all_tensors(get_model_weights_map(model.name), mem);
    model.close_model_file();
    model.tokenizer.load(model.tokenizer_data);

    // create thread pool
    auto *pool = nnml_threadpool_new(
        n_nodes,
        n_thread_per_node,
        false,
        affinity_mode,
        nnml_single_graph_compute_thread
    );
    if (is_print_binding) nnml_print_cpu_bindings(pool);
    pool->attach_work_buffer(temp_work_size, temp_work_data);
    pool->set_view_groups(1);

    // kvcache and cgraph
    uint32_t n_pad = is_fattn ? 256u : 32u;
    llm_kv_cache * kvcache = new llm_kv_cache(type_k, type_v, false, true, max_length, 1, n_pad,
                                              model.hparams, mem, n_nodes, is_print_kv);
    model.cgraph = new nnml_cgraph(model.get_n_tensors(), model.hparams, mem, is_tp, n_nodes,
                                   model.name, is_fattn, false, true, kvcache);

    // decoding context
    decoding_context ctx;
    g_ctx = &ctx;
    ctx.cparams = llm_cparams(max_length, 1024, 256, 1, n_nodes * n_thread_per_node,
                              10000.0f, 1.0f, is_fattn, !is_print_perf, true);
    ctx.scheduler = new nnml_scheduler();
    std::vector<nnml_cgraph *> graphs;
    graphs.push_back(model.cgraph);
    ctx.scheduler->init(pool, graphs);
    ctx.is_tp = is_tp;
    ctx.cgraph = model.cgraph;
    ctx.model = &model;
    ctx.balloc = new llm_batch_allocr(model.hparams.n_pos_per_embd());
    ctx.samp_params = sampler_params();
    ctx.output_ids.resize(max_length);

    // multiple rounds of chat
    std::vector<chat_msg> msgs;

    if (!prompt.empty()) {
        msgs.push_back(chat_msg("user", prompt));
        printf("> %s\n", prompt.c_str());
    }

    auto interrupt_callback = [&]() {
        if (g_interrupted) return false;
        return true;
    };
    ctx.token_callback = interrupt_callback;

    while (true) {
        if (msgs.empty()) {
            std::string user_input;
            printf("> ");

            if (!std::getline(std::cin, user_input)) break;
            if (user_input.empty()) continue;

            msgs.push_back(chat_msg("user", user_input));
        }

        std::string prompt_in = apply_template(model.name, msgs, true, false);
        std::vector<llm_token> encoded = common_tokenize(&model.tokenizer, prompt_in, false, true);

        g_exit = true;
        g_interrupted = false;
        
        ctx.infer(encoded, max_gen);

        g_exit = false;
        prompt.clear();
        msgs.clear();
    }

    return 0;
}