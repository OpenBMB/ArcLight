/**
 * @file sampler.h
 * @brief mini sampling functions for LLM inference.
 * 
 * Provides:
 *  - Definition of the `sampler_type` enum for different sampling strategies.
 *  - Definition of the `sampler_params` struct for configuring sampling parameters.
 *  - Declaration of the `sample_token` function for sampling a token from logits.
 * 
 * Designed for sampling tokens during LLM inference, supporting various strategies like top-k, top-p, and temperature sampling.
 */
#pragma once

#include <string>
#include <vector>
#include <set>

#include "tensor.h"

#define LLM_DEFAULT_SEED 0xFFFFFFFF

enum sampler_type {
    SAMPLER_TYPE_NONE        = 0,
    SAMPLER_TYPE_TOP_K       = 1,
    SAMPLER_TYPE_TOP_P       = 2,
    SAMPLER_TYPE_TEMPERATURE = 3,
    SAMPLER_TYPE_COUNT
};

struct sampler_params {
    int   top_k         = 40;
    float top_p         = 0.9f;
    float temperature   = 1.0f;
};


/**
 * sample_token: the mini sampling interface for LLM inference.
 */
int sample_token(const nnml_tensor * logits,
                 sampler_type type,
                 const sampler_params & params);
