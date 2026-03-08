#pragma once

#include <map>

#include "cgraph.h"
#include "model.h"


// hparams map register and query
// model definers should register their model hparam map as follows: (put this line in your xxx.cpp file)
// REGISTER_HPARAMS_MAP(xxx, params_map);
#define REGISTER_HPARAMS_MAP(key, map)     static bool _unused_hp_##key = (get_hparams_map()[#key] = map, true)
inline std::map<std::string, std::map<std::string, llm_hparams_item>> & get_hparams_map() {
    static std::map<std::string, std::map<std::string, llm_hparams_item>> hparams_maps;
    return hparams_maps;
}
// query
inline std::map<std::string, llm_hparams_item> & get_model_hparams_map(const std::string & model_name) {
    if (get_hparams_map().count(model_name)) return get_hparams_map()[model_name];
    else abort();
}


// weights map register and query
// model definers should register their model weights map as follows: (put this line in your xxx.cpp file)
// REGISTER_WEIGHTS_MAP(xxx, weights_map);
#define REGISTER_WEIGHTS_MAP(key, map)     static bool _unused_wt_##key = (get_weights_map()[#key] = map, true)
inline std::map<std::string, std::map<std::string, llm_weight_item>> & get_weights_map() {
    static std::map<std::string, std::map<std::string, llm_weight_item>> weights_maps;
    return weights_maps;
}
// query
inline std::map<std::string, llm_weight_item> & get_model_weights_map(const std::string & model_name) {
    if (get_weights_map().count(model_name)) return get_weights_map()[model_name];
    else abort();
}


// hyperparameters register and query
// model definers should register their model's hyperparameters as follows: (put this line in your xxx.cpp file)
// REGISTER_ROPE_TYPE(xxx, llm_rope_type::LLM_ROPE_TYPE_XXX);
// REGISTER_FREQ_BASE(xxx, 100000.0f);
#define REGISTER_ROPE_TYPE(key, value)     static bool _unused_rt_##key = (get_rope_type()[#key] = value, true)
#define REGISTER_FREQ_BASE(key, value)     static bool _unused_fb_##key = (get_freq_base()[#key] = value, true)
#define REGISTER_FREQ_SCALE(key, value)    static bool _unused_fs_##key = (get_freq_scale()[#key] = value, true)
#define REGISTER_EXT_FACTOR(key, value)    static bool _unused_ef_##key = (get_ext_factor()[#key] = value, true)
#define REGISTER_ATTN_FACTOR(key, value)   static bool _unused_af_##key = (get_attn_factor()[#key] = value, true)
#define REGISTER_BETA_FAST(key, value)     static bool _unused_bf_##key = (get_beta_fast()[#key] = value, true)
#define REGISTER_BETA_SLOW(key, value)     static bool _unused_bs_##key = (get_beta_slow()[#key] = value, true)
inline std::map<std::string, llm_rope_type>& get_rope_type() {
    static std::map<std::string, llm_rope_type> rope_types;
    return rope_types;
}
inline std::map<std::string, float>& get_freq_scale() {
    static std::map<std::string, float> freq_scales;
    return freq_scales;
}
inline std::map<std::string, float>& get_freq_base() {
    static std::map<std::string, float> freq_bases;
    return freq_bases;
}
inline std::map<std::string, float>& get_ext_factor() {
    static std::map<std::string, float> ext_factors;
    return ext_factors;
}
inline std::map<std::string, float>& get_attn_factor() {
    static std::map<std::string, float> attn_factors;
    return attn_factors;
}
inline std::map<std::string, float>& get_beta_fast() {
    static std::map<std::string, float> beta_fasts;
    return beta_fasts;
}
inline std::map<std::string, float>& get_beta_slow() {
    static std::map<std::string, float> beta_slows;
    return beta_slows;
}
// query
inline llm_rope_type get_model_rope_type(const std::string & model_name) 
    { return get_rope_type().count(model_name) ? get_rope_type()[model_name] : llm_rope_type::LLM_ROPE_TYPE_NONE; }
inline float get_model_freq_base(const std::string & model_name) 
    { return get_freq_base().count(model_name) ? get_freq_base()[model_name] : 100000.0f; }
inline float get_model_freq_scale(const std::string & model_name) 
    { return get_freq_scale().count(model_name) ? get_freq_scale()[model_name] : 1.0f; }
inline float get_model_ext_factor(const std::string & model_name) 
    { return get_ext_factor().count(model_name) ? get_ext_factor()[model_name] : 0.0f; }
inline float get_model_attn_factor(const std::string & model_name) 
    { return get_attn_factor().count(model_name) ? get_attn_factor()[model_name] : 1.0f; }
inline float get_model_beta_fast(const std::string & model_name) 
    { return get_beta_fast().count(model_name) ? get_beta_fast()[model_name] : 32.0f; }
inline float get_model_beta_slow(const std::string & model_name) 
    { return get_beta_slow().count(model_name) ? get_beta_slow()[model_name] : 1.0f; }


// templates register and caller
// model definers should register their model's template caller as follows: (put this line in your xxx.cpp file)
// REGISTER_TEMPLATE_CALLER(xxx, xxx_apply_template);
using template_caller = std::function<std::string(const std::vector<chat_msg> & messages, bool add_generation_prompt, bool enable_reasoning)>;
#define REGISTER_TEMPLATE_CALLER(key, func) static bool _unused_tc_##name = (get_template_callers()[#key] = func, true)
inline std::map<std::string, template_caller> & get_template_callers() {
    static std::map<std::string, template_caller> template_callers;
    return template_callers;
}
// call
inline std::string apply_template(const std::string & model_name, const std::vector<chat_msg> & messages, bool add_generation_prompt, bool enable_reasoning) {
    if (get_template_callers().count(model_name)) {
        return get_template_callers()[model_name](messages, add_generation_prompt, enable_reasoning);
    } else abort();
}


// model definition
// model definers should define your model as follows: (put this line in your xxx.cpp file)
// void build_xxx_forward(nnml_cgraph & graph, llm_model & model, bool is_tp);


// model register
// model definers should register your model as follows: (put this line in your xxx.cpp file)
// static bool _reg_xxx_builder = (nnml_cgraph::reg_builder("xxx", build_xxx_forward), true);


inline void append_block(std::string & out, const std::string & role, const std::string & content)
{
    out.append("<|im_start|>");
    out.append(role);
    out.append("\n");
    out.append(content);
    out.append("<|im_end|>\n");
}
