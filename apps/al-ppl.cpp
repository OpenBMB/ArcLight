#include <stdio.h>
#include <math.h>
#include <float.h>

#include "arg.h"
#include "model.h"
#include "models.h"
#include "context.h"
#include "ops.h"
#include "scheduler.h"


arg_parser args;

void softmax_inplace(float * x, int n) {
    float maxv = -FLT_MAX;
    for (int i = 0; i < n; ++i) { if (x[i] > maxv) maxv = x[i]; }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}

float compute_perplexity(const std::vector<llm_token> & tokens, nnml_tensor * t_logits) {
    uint32_t n_cols = t_logits->get_elements(0);
    uint32_t n_rows = t_logits->get_elements(1);
    // here we assume that the logits from nnml is always with F32 type
    float loss = 0.0f;
    for (uint32_t i = 1; i < n_rows; ++i) {
        if (tokens[i] >= n_cols) {
            throw std::runtime_error("token id out of range in logits");
        }

        uint32_t base = (i - 1) * n_cols;
        uint32_t idx = (i - 1) * n_cols + tokens[i];

        softmax_inplace((float*)t_logits->tensor_data() + base, n_cols);
        
        float logit = ((float*)t_logits->tensor_data())[idx];
        loss -= logf(logit);
        // printf("idx: %d, token_id: %d, logit: %.4f\n", idx, tokens[i], logit);
    }
    // printf("total loss: %.4f\n", loss);
    return expf(loss / (n_rows - 1));
}

int32_t main(int32_t argc, char** argv) {
    // print_logo();
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

    // printf("current args: asm=%d, fattn=%d, print_model=%d, print_binding=%d, print_kv=%d, threads=%d, nodes=%d, "
    //        "w_gb=%d, a_gb=%d, kv_gb=%d, work_gb=%d, max_length=%d, max_gen=%d, model=%s, numa=%s, prompt=%s, tk=%d, tv=%d\n",
    //        is_asm, is_fattn, is_print_model, is_print_binding, is_print_kv, n_threads, n_nodes,
    //        w_gb, a_gb, kv_gb, work_gb, max_length, max_gen, model_path.c_str(), numa_strategy.c_str(),
    //        prompt.c_str(), type_k, type_v);

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
                                   model.name, is_fattn, false, true, kvcache, true);
    
    // prepare input
    printf("Evaluating PPL on text: %s\n", prompt.c_str());
    std::vector<llm_token> encoded = common_tokenize(&model.tokenizer, prompt, false, true);
    uint32_t eval_length = (encoded.size() + 15) & ~15;
    
    // decoding context
    // it requests that n_ubatch is not less than eval_length in ppl evaluation
    decoding_context ctx;
    ctx.cparams = llm_cparams(max_length, eval_length, eval_length, 1, n_nodes * n_thread_per_node,
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

    ctx.infer(encoded, 0);
    float ppl = compute_perplexity(encoded, ctx.res_logits);
    printf("perplexity: %.2f\n", ppl);

    return 0;
}
