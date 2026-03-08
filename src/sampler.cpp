/**
 * @file sampler.cpp
 * @brief Implementation of sampler.h
 * 
 * This file implements the functions declared in sampler.h for sampling tokens from logits during LLM inference.
 * We use a simple design in the current implementation.
 */
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

#include "sampler.h"


// we have not implemented batch sampling yet, so we use a single global RNG
// in the future, we may want to have per-context RNGs
static std::mt19937 rng(std::random_device{}());

static std::vector<float> tensor_to_vector(const nnml_tensor * t) {
    int n = t->n_elements();
    const float * ptr = (const float *) t->tensor_data();
    return std::vector<float>(ptr, ptr + n);
}

static void softmax_inplace(std::vector<float> & x) {
    float maxv = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float &v : x) { v = std::exp(v - maxv); sum += v; }
    for (float &v : x) v /= sum;
}

static int sample_none(const std::vector<float> & logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

static int sample_temperature(std::vector<float> logits, float temp) {
    for (float &x : logits) x /= temp;
    softmax_inplace(logits);

    std::discrete_distribution<int> dist(logits.begin(), logits.end());
    return dist(rng);
}

static int sample_top_k(std::vector<float> logits, int top_k) {
    int n = logits.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    std::vector<float> probs;
    std::vector<int>   tokens;
    probs.reserve(top_k);
    tokens.reserve(top_k);

    for (int i = 0; i < top_k; ++i) {
        tokens.push_back(idx[i]);
        probs.push_back(logits[idx[i]]);
    }

    softmax_inplace(probs);

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return tokens[dist(rng)];
}

static int sample_top_p(std::vector<float> logits, float top_p) {
    int n = logits.size();

    // softmax
    softmax_inplace(logits);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });

    float cumsum = 0.f;
    int cutoff = 0;
    for (; cutoff < n; ++cutoff) {
        cumsum += logits[idx[cutoff]];
        if (cumsum >= top_p) break;
    }
    cutoff = std::max(1, cutoff + 1);

    std::vector<float> probs;
    std::vector<int> tokens;
    probs.reserve(cutoff);
    tokens.reserve(cutoff);

    for (int i = 0; i < cutoff; ++i) {
        tokens.push_back(idx[i]);
        probs.push_back(logits[idx[i]]);
    }

    // softmax
    softmax_inplace(probs);

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return tokens[dist(rng)];
}

int sample_token(const nnml_tensor * logits,
                 sampler_type type,
                 const sampler_params & params)
{
    std::vector<float> v = tensor_to_vector(logits);

    switch (type) {
        case SAMPLER_TYPE_NONE:
            return sample_none(v);
        case SAMPLER_TYPE_TEMPERATURE:
            return sample_temperature(v, params.temperature);
        case SAMPLER_TYPE_TOP_K:
            return sample_top_k(v, params.top_k);
        case SAMPLER_TYPE_TOP_P:
            return sample_top_p(v, params.top_p);
        default:
            return sample_none(v);      // greedy
    }
}
