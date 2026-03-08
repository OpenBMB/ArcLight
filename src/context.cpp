/**
 * @file context.cpp
 * @brief Implementation of context.h
 * 
 * This file implements the functions declared in context.h.
 *  - infer -> decode -> process_ubatch
 *  - inder: the overall decoding loop, from prompt to generation.
 *  - decode: one forward step of decoding, from batch preparation to running the computation graph.
 *  - process_ubatch: process one ubatch, including judging whether to rebuild the graph.
 */
#include <cassert>
#include <cstring>
#include <algorithm>
#include <sstream>

#include "context.h"
#include "tokenizer.h"


// implementation of decoding_context
void decoding_context::infer(std::vector<llm_token> & input_tokens, int max_new_tokens) {
    int64_t t_prefill_start_us = nnml_time_us();
    int n_past = 0;
    for (int i = 0; i < (int) input_tokens.size(); i += cparams.n_batch) {
        int n_eval = (int) input_tokens.size() - i;
        if (n_eval > cparams.n_batch) {
            n_eval = cparams.n_batch;
        }
        n_past += n_eval;
        
        if (decode(llm_batch_get_one(&input_tokens[i], n_eval))) {
            LLM_ERROR("%s : failed to prefill\n", __func__);
            return;
        }
    }
    int64_t t_prefill_end_us = nnml_time_us();
    t_prefill_us += t_prefill_end_us - t_prefill_start_us;
    n_prefill_tokens += n_past;
    
    assert(n_past == (int) input_tokens.size());
    // input_tokens.clear();
    int r_token, n_steps = 0;
    r_token = sample_token(res_logits, samp_type, samp_params);
    if (!llm_vocab_is_eog(&(model->tokenizer), (llm_token)r_token) && max_new_tokens > 0) {
        const std::string token_str = common_token_to_piece(&(model->tokenizer), r_token, true);
        LLM_LOG(true, "%s", token_str.c_str());
        fflush(stdout);
    }

    int64_t t_decode_start_us = nnml_time_us();
    while (!llm_vocab_is_eog(&(model->tokenizer), (llm_token)r_token) && n_steps < max_new_tokens) {
        n_steps++;
        if (decode(llm_batch_get_one((llm_token *)&r_token, 1))) {
            LLM_ERROR("%s : failed to decode\n", __func__);
            return;
        }

        r_token = sample_token(res_logits, samp_type, samp_params);
        if (token_callback && !token_callback()) {                      // interrupt the decoding loop if the callback returns false
            break;
        }
        if (!llm_vocab_is_eog(&(model->tokenizer), (llm_token)r_token)) {
            const std::string token_str = common_token_to_piece(&(model->tokenizer), r_token, true);
            LLM_LOG(true, "%s", token_str.c_str());
            fflush(stdout);
        }
    }
    LLM_LOG(true, "\n");
    int64_t t_decode_end_us = nnml_time_us();
    t_decode_us += t_decode_end_us - t_decode_start_us;
    n_decode_tokens += n_steps;
}

int decoding_context::decode(const llm_batch & batch_inp) {
    assert((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd));
    if (batch_inp.n_tokens == 0) {
        LLM_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }
    const auto & vocab   = model->tokenizer;
    const auto & hparams = model->hparams;
    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd;

    if (init_batch(batch_inp, n_embd, cparams.kv_unified ? LLM_MAX_SEQ : cparams.n_seq_max, false) == false) {
        LLM_ERROR("%s: failed to initialize batch allocator\n", __func__);
        return -1;
    }
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    
    bool did_optimize = false;
    
    if (!init_ubatch(cparams.n_ubatch)) {
        LLM_ERROR("%s: failed to initialize ubatch\n", __func__);
        return -1;
    }

    do {
        const auto & ubatch = cgraph->get_kvcache()->get_ubatch();
        // printf("decoding_context::decode: processing ubatch with %u tokens\n", ubatch.n_tokens);
        bool res = process_ubatch(ubatch);
        // printf("after process_ubatch ok~\n");
        if (!res) {
            LLM_ERROR("%s: failed to process ubatch\n", __func__);
            return -1;
            llm_pos pos_min[LLM_MAX_SEQ];
            for (int s = 0; s < LLM_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llm_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLM_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llm_pos>::max()) {
                    continue;
                }

                LLM_LOG(true, "%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                cgraph->get_kvcache()->seq_rm(s, pos_min[s], -1);
            }
        }
        res_logits = cgraph->get_logits();
        
    } while (cgraph->get_kvcache()->next());

    uint32_t n_outputs = n_outputs_all;
    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;
        auto & out_ids = balloc->get_out_ids();
        assert(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output) {
            assert((size_t) n_outputs == out_ids.size());

            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);
                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    return 0;
}

bool decoding_context::process_ubatch(const llm_ubatch & ubatch) {
    if (cgraph && cgraph->get_kvcache() && !cgraph->get_kvcache()->apply()) {
        LLM_ERROR("%s: failed to apply kvcache context\n", __func__);
        return false;
    }
    
    if (cgraph->get_nodes_ptr() != nullptr && 
        ubatch.n_tokens == cgraph->get_n_tokens() &&
        cgraph->get_n_kv_max() >= cgraph->get_kvcache()->get_n_kv()) {
        n_reused++;
    } else {
        // cgraph->reset
        cgraph->graph_clear();
        cgraph->set_n_tokens(ubatch.n_tokens);
        cgraph->set_ubatch(ubatch);
        // printf("process_ubatch: rebuilding computation graph for ubatch with %u tokens\n", cgraph->n_tokens);
        n_reused = 0;

        cgraph->get_mem()->clear_activation_buffers();
        // cgraph->mem->clear_kv_buffers();

        cgraph->build_model_forward(*model, is_tp);

    }
    cgraph->set_inputs(&ubatch);
    
    scheduler->nnml_single_graph_compute();
    return true;
}

bool decoding_context::init_batch(const llm_batch & batch_inp, uint32_t n_embd, uint32_t n_seq_max, bool output_all) {
    return balloc->init(batch_inp, model->tokenizer, cgraph->get_kvcache(), n_embd, n_seq_max, output_all);
}

bool decoding_context::init_ubatch(uint32_t n_ubatch) {
    do {
        balloc->split_reset();
        std::vector<llm_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc->split_simple(n_ubatch) : balloc->split_equal(n_ubatch, true);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }
        if (balloc->get_n_used() < balloc->get_n_tokens()) {
            // failed to find a suitable split
            break;
        }
        auto sinfos = cgraph->get_kvcache()->prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        cgraph->get_kvcache()->set_ubatchs(ubatches);
        cgraph->get_kvcache()->set_sinfos(sinfos);
        cgraph->get_kvcache()->set_i_cur(0);

        return true;
    } while (false);

    return false;
}

llm_ubatch decoding_context::get_ubatch() {
    return cgraph->get_kvcache()->get_ubatch();
}

void decoding_context::print_perf() const {
    if (cparams.no_perf) {
        return;
    }
    LLM_LOG(true, "prefill time: %.2f ms / %4d token, %.2f ms/token, %.2f token/s\n", 
            t_prefill_us / 1000.0, n_prefill_tokens, 
            t_prefill_us / 1000.0 / n_prefill_tokens, n_prefill_tokens * 1000000.0 / t_prefill_us);
    LLM_LOG(true, "decode time:  %.2f ms / %4d token, %.2f ms/token, %.2f token/s\n", 
            t_decode_us / 1000.0, n_decode_tokens, 
            t_decode_us / 1000.0 / n_decode_tokens, n_decode_tokens * 1000000.0 / t_decode_us);
}


// implementation of llm_batch_allocr
llm_batch_allocr::llm_batch_allocr(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {
    // const char * LLM_BATCH_DEBUG = getenv("LLM_BATCH_DEBUG");
    // debug = LLM_BATCH_DEBUG ? atoi(LLM_BATCH_DEBUG) : 0;
    debug = 0;
    seq_pos.resize(LLM_MAX_SEQ);
    seq_cpl.resize(LLM_MAX_SEQ);
    for (auto & cur : seq_cpl) cur.resize(LLM_MAX_SEQ);
    seq_idx.resize(LLM_MAX_SEQ, -1);
}

bool llm_batch_allocr::init(
        const llm_batch & batch_inp,
        const llm_vocab & vocab,
        const llm_kv_cache * kvcache,
        uint32_t n_embd,
        uint32_t n_seq_max,
        bool output_all) {
    clear();
    batch = batch_inp;
    this->vocab = &vocab;
    assert(batch.n_tokens > 0);
    //
    // validate input batch
    //
    if (n_seq_max > LLM_MAX_SEQ) {
        LLM_ERROR("%s: n_seq_max = %d > %d\n", __func__, n_seq_max, LLM_MAX_SEQ);
        return false;
    }
    if (batch.token) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= vocab.n_tokens()) {
                LLM_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return false;
            }
        }
    }
    if (batch.seq_id) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                if (batch.seq_id && (batch.seq_id[i][s] < 0 || batch.seq_id[i][s] >= (llm_seq_id) n_seq_max)) {
                    LLM_ERROR("%s: invalid seq_id[%d][%d] = %d >= %d\n", __func__, i, s, batch.seq_id[i][s], (llm_seq_id) n_seq_max);
                    return false;
                }
            }
        }
    }

    //
    // auto-generate missing fields
    //

    if (!batch.n_seq_id) {
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size();
        }
        batch.n_seq_id = n_seq_id.data();
    }

    if (!batch.seq_id) {
        seq_id.resize(batch.n_tokens + 1);
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }

    if (!batch.pos) {
        pos.resize(batch.n_tokens);
        // initialize the starting position for each sequence based on the positions in the memory
        llm_pos p0[LLM_MAX_SEQ];
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (!kvcache) {
                // if no memory -> start from 0
                p0[s] = 0;
            } else {
                p0[s] = kvcache->seq_pos_max(s) + 1;
            }
        }
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            const llm_seq_id seq_id = batch.seq_id[i][0];
            pos[i] = p0[seq_id];
            // update the starting position for all sequences that are assigned to the this token
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llm_seq_id seq_id = batch.seq_id[i][s];
                p0[seq_id] = pos[i] + 1;
            }
        }
        batch.pos = pos.data();
    }

    if (!batch.logits) {
        if (output_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (output_all) {
        bool warn = false;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] == 0) {
                warn = true;
            }
        }

        if (warn) {
            LLM_LOG(true, "%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", __func__);

            output.resize(batch.n_tokens, true);
            batch.logits = output.data();
        }
    }

    //
    // compute stats
    //

    this->n_embd    = n_embd;
    this->n_seq_max = n_seq_max;
    // count the outputs in this batch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }

    has_cpl = false;

    // determine coupled sequences
    // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llm_seq_id s0 = batch.seq_id[i][0];
        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llm_seq_id s1 = batch.seq_id[i][s];
            seq_pos[s1].insert(batch.pos[i]);
            if (s > 0) {
                // mark that sequence s1 is coupled to s0
                seq_cpl[s1][s0] = true;
                // note: tracking the other way around is not necessary for now
                //seq_cpl[s0][s1] = true;
                has_cpl = true;
            }
        }
    }

    // precompute the sequence sets for each token and determine the unique sequence ids that participate in the batch
    {
        seq_set_t seq_set_unq;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            seq_set_t cur;
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llm_seq_id seq_id = batch.seq_id[i][s];

                cur        .set(seq_id);
                seq_set_unq.set(seq_id);
            }

            seq_set.push_back(cur);
            seq_set_map[cur].push_back(i);
        }

        for (uint32_t s = 0; s < n_seq_max; ++s) {
            if (seq_set_unq.test(s)) {
                seq_idx[s] = seq_id_unq.size();
                seq_id_unq.push_back(s);
            }
        }
    }

    if (debug > 0) {
        LLM_LOG(true, "%s: input batch info:\n", __func__);

        llm_ubatch ubatch {
            /*.b_equal_seqs =*/ false,
            /*.n_tokens     =*/ (uint32_t) batch.n_tokens,
            /*.n_seq_tokens =*/ (uint32_t) 1,
            /*.n_seqs       =*/ (uint32_t) batch.n_tokens,
            /*.n_seqs_unq   =*/ (uint32_t) this->seq_id_unq.size(),
            /*.token        =*/ batch.token,
            /*.embd         =*/ batch.embd,
            /*.pos          =*/ batch.pos,
            /*.n_seq_id     =*/ batch.n_seq_id,
            /*.seq_id       =*/ batch.seq_id,
            /*.seq_id_unq   =*/ this->seq_id_unq.data(),
            /*.seq_idx      =*/ this->seq_idx.data(),
            /*.output       =*/ batch.logits,
            /*.data         =*/ {},
        };

        ubatch_print(ubatch, debug);

        LLM_LOG(true, "%s:   seq       = [\n", __func__);
        for (int s0 = 0; s0 < (int) seq_pos.size(); ++s0) {
            if (seq_pos[s0].empty()) {
                continue;
            }

            std::stringstream ss;
            for (int s1 = 0; s1 < (int) seq_cpl[s0].size(); ++s1) {
                if (seq_cpl[s0][s1]) {
                    ss << s1 << " ";
                }
            }

            LLM_LOG(true, "%s:  %4d: pos = [%4d, %4d], cpl = %s\n",
                    __func__, s0, seq_pos_min(s0), seq_pos_max(s0), ss.str().empty() ? "-" : ss.str().c_str());
        }
        LLM_LOG(true, "%s:   ]\n", __func__);
    }

    //
    // consistency checks
    //

    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_pos[s].empty()) {
            continue;
        }
        const llm_pos p0 = kvcache ? kvcache->seq_pos_max(s) : -1;
        if (p0 >= 0) {
            bool ok = true;
            if (batch.token) {
                if (seq_pos_min(s) != p0 + 1) {
                    ok = false;
                }
            } else {
                assert(batch.embd);
                // for embeddings (typically used as vision input), we allow them to have repeating positions
                if (seq_pos_min(s) != p0 && seq_pos_min(s) != p0 + 1) {
                    ok = false;
                }
            }

            if (!ok) {
                LLM_ERROR(
                        "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                        " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n"
                        " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                        " it is required that the sequence positions remain consecutive: Y = X + 1\n",
                        __func__, s, s, p0, s, seq_pos_min(s));

                return false;
            }
        }

        if (seq_pos_max(s) - seq_pos_min(s) + 1 > (int) seq_pos[s].size()) {
            LLM_ERROR("%s: sequence %d positions are not continuous\n", __func__, s);
            return false;
        }
    }

    if (kvcache) {
        for (uint32_t s0 = 0; s0 < n_seq_max; ++s0) {
            for (uint32_t s1 = 0; s1 < n_seq_max; ++s1) {
                if (seq_cpl[s0][s1]) {
                    if (kvcache->seq_pos_min(s0) != kvcache->seq_pos_min(s1) ||
                        kvcache->seq_pos_max(s0) != kvcache->seq_pos_max(s1)) {
                        LLM_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, s0, s1);
                        return false;
                    }
                }
            }
        }
    }

    // disallow partial sequence sub-sets:
    //
    // invalid:          x
    //            i: 0 1 2 ...
    // ---------------------------------------
    // seq_id[i][0]: 0 0 1
    // seq_id[i][1]: 1 1 2
    // seq_id[i][2]: 2
    //
    // disallow decreasing sequence positions:
    //
    // invalid:                  x
    //            i: 0 1 2 3 4 5 6 ...
    // ---------------------------------------
    //       pos[i]: 4 5 0 1 6 2 3
    // seq_id[i][0]: 0 0 1 1 0 1 0
    //
    {
        seq_set_t cur_seq_set[LLM_MAX_SEQ];
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            cur_seq_set[s].set();
        }

        llm_pos cur_seq_pos[LLM_MAX_SEQ];
        for (uint32_t s = 0; s < n_seq_max; ++s) {
            cur_seq_pos[s] = -1;
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const llm_pos pos = batch.pos[i];

            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llm_seq_id seq_id = batch.seq_id[i][s];
                cur_seq_set[seq_id] &= seq_set[i];
                if (cur_seq_set[seq_id].none()) {
                    LLM_ERROR("%s: sequence %d belongs to incompatible sequence sets (not allowed)\n", __func__, seq_id);
                    return false;
                }
                if (pos < cur_seq_pos[seq_id]) {
                    LLM_ERROR("%s: sequence %d positions are decreasing (not allowed)\n", __func__, seq_id);
                    return false;
                }
            }
        }
    }

    split_reset();

    return true;
}

llm_ubatch llm_batch_allocr::ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs) {
    const uint32_t n_tokens = n_seq_tokens*n_seqs;

    clear();
    split_reset();

    auto udata = std::make_shared<llm_ubatch::data_t>();
    udata->token     .resize(n_tokens);
    udata->embd      .clear();
    udata->pos       .resize(n_tokens);
    udata->n_seq_id  .resize(n_tokens);
    udata->seq_id    .resize(n_tokens);
    udata->seq_id_unq.resize(0);
    udata->seq_idx   .resize(LLM_MAX_SEQ, -1);
    udata->output    .resize(n_tokens);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        udata->seq_idx[s] = s;
        udata->seq_id_unq.push_back(s);
    }

    llm_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ n_seqs,

        /*.token        =*/ udata->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
        /*.seq_id_unq   =*/ udata->seq_id_unq.data(),
        /*.seq_idx      =*/ udata->seq_idx.data(),
        /*.output       =*/ udata->output.data(),
        /*.data         =*/ std::move(udata),
    };

    return res;
}

const llm_batch & llm_batch_allocr::get_batch() const {
    return batch;
}

uint32_t llm_batch_allocr::get_n_tokens() const {
    return batch.n_tokens;
}

uint32_t llm_batch_allocr::get_n_outputs() const {
    return n_outputs;
}

uint32_t llm_batch_allocr::get_n_used() const {
    return n_used;
}

std::vector<int32_t> & llm_batch_allocr::get_out_ids() {
    return out_ids;
}

llm_pos llm_batch_allocr::seq_pos_min(llm_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].begin();
}

llm_pos llm_batch_allocr::seq_pos_max(llm_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].rbegin();
}

void llm_batch_allocr::split_reset() {
    out_ids.clear();
    n_used = 0;
    used.clear();
    used.resize(get_n_tokens(), false);
}

llm_ubatch llm_batch_allocr::split_simple(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }
    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }
    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);
        used[cur_idx] = true;
        ++n_used;
        ++cur_idx;
        if (cur_idx >= used.size()) {
            break;
        }
        if (idxs.size() >= n_ubatch) {
            break;
        }
    }
    // printf("split_simple: created ubatch with %zu tokens\n", idxs.size());
    return ubatch_add(idxs, idxs.size(), false);
}

llm_ubatch llm_batch_allocr::split_equal(uint32_t n_ubatch, bool sequential) {
    if (sequential && has_cpl) {
        LLM_ERROR("%s: sequential split is not supported when there are coupled sequences in the input batch (you may need to use the -kvu flag)\n", __func__);

        return {};
    }
    std::vector<seq_set_t> cur_seq_set;
    llm_seq_id last_seq_id = -1;
    // determine the non-overlapping sequence sets participating in this ubatch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) {
            continue;
        }
        bool add = true;

        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            // no overlap with existing sequence sets:
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }

        // accept only increasing sequence ids
        if (sequential) {
            add = add && (cur_seq_set.empty() || batch.seq_id[i][0] == last_seq_id + 1);
        }

        if (add) {
            cur_seq_set.push_back(seq_set[i]);
            last_seq_id = batch.seq_id[i][0];
            if (cur_seq_set.size() > n_ubatch) {
                break;
            }
        }
    }

    const uint32_t n_seqs = cur_seq_set.size();
    // we are done
    if (n_seqs == 0) {
        return {};
    }

    // the current batch index of each sequence set
    std::vector<int32_t> cur_idx(n_seqs, 0);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        while (used[seq_set_map[cur_seq_set[s]][cur_idx[s]]]) {
            ++cur_idx[s];
        }
    }
    // the list of batch indices for each sequence set
    // at the end we will concat these to get the final ubatch
    std::vector<idx_vec_t> idxs_per_seq(n_seqs);

    while (true) {
        // we can only add new n_seq_tokens tokens if all the sequence sets have at least one more unused token and
        //   if we haven't reached n_ubatch
        bool can_expand = true;

        for (uint32_t s = 0; s < n_seqs; ++s) {
            if (cur_idx[s] >= (int32_t) seq_set_map[cur_seq_set[s]].size()) {
                can_expand = false;
                break;
            }
        }
        if (!can_expand) {
            break;
        }
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const int32_t idx = seq_set_map[cur_seq_set[s]][cur_idx[s]];

            idxs_per_seq[s].push_back(idx);

            used[idx] = true;
            ++n_used;

            ++cur_idx[s];
        }

        if  ((idxs_per_seq[0].size() + 1)*n_seqs > n_ubatch) {
            break;
        }
    }

    // concat the per-sequence-set lists
    std::vector<int32_t> idxs;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end());
    }

    return ubatch_add(idxs, n_seqs, true);
}

llm_ubatch llm_batch_allocr::split_seq(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }

    // this is the starting sequence set
    // we allow adding tokens only if their sequence set is a subset of the current sequence set
    auto cur_seq_set = seq_set[cur_idx];

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;
        ++n_used;

        if (idxs.size() >= n_ubatch) {
            break;
        }

        do {
            ++cur_idx;
        } while (cur_idx < get_n_tokens() && (used[cur_idx] || ((cur_seq_set & seq_set[cur_idx]) != seq_set[cur_idx])));

        if (cur_idx == get_n_tokens()) {
            break;
        }

        cur_seq_set = seq_set[cur_idx];
    }

    return ubatch_add(idxs, 1, true);
}

void llm_batch_allocr::clear() {
    n_outputs = 0;
    batch = {};
    pos       .clear();
    n_seq_id  .clear();
    seq_id    .clear();
    seq_id_unq.clear();
    output    .clear();

    for (auto & cur : seq_pos) {
        cur.clear();
    }

    for (auto & cur : seq_cpl) {
        std::fill(cur.begin(), cur.end(), false);
    }

    seq_set.clear();

    seq_set_map.clear();

    std::fill(seq_idx.begin(), seq_idx.end(), -1);
}

llm_ubatch llm_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    const uint32_t n_tokens = idxs.size();

    assert(n_tokens%n_seqs == 0);

    auto udata = std::make_shared<llm_ubatch::data_t>();

    const int32_t n_pos_cur = batch.embd ? n_pos_per_embd : 1;

    const int64_t n_embd_all = batch.embd ? (int64_t) n_tokens*n_embd : 0;
    const int64_t n_pos_all  =              (int64_t) n_tokens*n_pos_cur;

    udata->token     .resize(n_tokens);
    udata->embd      .resize(n_embd_all);
    udata->pos       .resize(n_pos_all);
    udata->n_seq_id  .resize(n_tokens);
    udata->seq_id    .resize(n_tokens);
    udata->seq_id_unq.resize(0);
    udata->seq_idx   .resize(LLM_MAX_SEQ, -1);
    udata->output    .resize(n_tokens);

    seq_set_t seq_set_unq;

    for (size_t i = 0; i < idxs.size(); ++i) {
        if (batch.token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (batch.embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        for (int j = 0; j < n_pos_cur; ++j) {
            udata->pos[j*n_tokens + i] = batch.pos[j*batch.n_tokens + idxs[i]];
        }

        udata->n_seq_id[i] = batch.n_seq_id[idxs[i]];
        udata->seq_id[i]   = batch.seq_id[idxs[i]];
        udata->output[i]   = batch.logits[idxs[i]];

        for (int s = 0; s < udata->n_seq_id[i]; ++s) {
            seq_set_unq.set(udata->seq_id[i][s]);
        }

        if (udata->output[i]) {
            out_ids.push_back(idxs[i]);
        }
    }

    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_set_unq.test(s)) {
            udata->seq_idx[s] = udata->seq_id_unq.size();
            udata->seq_id_unq.push_back(s);
        }
    }

    llm_ubatch res {
        /*.b_equal_seqs =*/ equal_seqs,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_tokens/n_seqs,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),

        /*.token        =*/ batch.token ? udata->token.data() : nullptr,
        /*.embd         =*/ batch.embd ? udata->embd.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
        /*.seq_id_unq   =*/ udata->seq_id_unq.data(),
        /*.seq_idx      =*/ udata->seq_idx.data(),
        /*.output       =*/ udata->output.data(),
        /*.data         =*/ std::move(udata),
    };
    // printf("ubatch_add: created ubatch with %d tokens, %d seq_tokens, %d seqs, %d seqs_unq\n",
    //        res.n_tokens, res.n_seq_tokens,  res.n_seqs, res.n_seqs_unq);

    if (debug > 0) {
        LLM_LOG(true, "%s: added ubatch to split:\n", __func__);

        ubatch_print(res, debug);
    }

    return res;
}

void llm_batch_allocr::ubatch_print(const llm_ubatch & ubatch, int debug) {
    if (debug > 0) {
        LLM_LOG(true, "%s:   equal_seqs   = %d\n", __func__, ubatch.equal_seqs());
        LLM_LOG(true, "%s:   n_tokens     = %d\n", __func__, ubatch.n_tokens);
        LLM_LOG(true, "%s:   n_seq_tokens = %d\n", __func__, ubatch.n_seq_tokens);
        LLM_LOG(true, "%s:   n_seqs       = %d\n", __func__, ubatch.n_seqs);
        LLM_LOG(true, "%s:   n_seqs_unq   = %d\n", __func__, ubatch.n_seqs_unq);

        std::stringstream ss_seq_id_unq;
        std::stringstream ss_seq_idx;

        ss_seq_id_unq << "[ ";
        ss_seq_idx << "[";

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            ss_seq_id_unq << ubatch.seq_id_unq[s] << " ";
        }

        for (uint32_t s = 0; s < LLM_MAX_SEQ; ++s) {
            if (ubatch.seq_idx[s] >= 0) {
                ss_seq_idx << ubatch.seq_idx[s]%10;
            } else {
                ss_seq_idx << ".";
            }
        }

        ss_seq_id_unq << "]";
        ss_seq_idx    << "]";

        LLM_LOG(true, "%s:   token      = %p\n", __func__, (void *) ubatch.token);
        LLM_LOG(true, "%s:   embd       = %p\n", __func__, (void *) ubatch.embd);
        LLM_LOG(true, "%s:   pos        = %p\n", __func__, (void *) ubatch.pos);
        LLM_LOG(true, "%s:   n_seq_id   = %p\n", __func__, (void *) ubatch.n_seq_id);
        LLM_LOG(true, "%s:   seq_id     = %p\n", __func__, (void *) ubatch.seq_id);
        LLM_LOG(true, "%s:   seq_id_unq = %s\n", __func__, ss_seq_id_unq.str().c_str());
        LLM_LOG(true, "%s:   seq_idx    = %s\n", __func__, ss_seq_idx.str().c_str());
        LLM_LOG(true, "%s:   output     = %p\n", __func__, (void *) ubatch.output);
        LLM_LOG(true, "%s:   n_outputs  = %d\n", __func__, n_outputs);

        if (debug > 1) {
            int seq_id_max = 0;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                        seq_id_max = std::max(seq_id_max, ubatch.seq_id[i][s]);
                    }
                }
            }
            ++seq_id_max;

            LLM_LOG(true, "%s:   token     = [\n", __func__);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                std::vector<int8_t> seq_id(seq_id_max);

                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    seq_id[ubatch.seq_id[i][s]] = 1;
                }

                std::stringstream ss;
                for (int s = 0; s < seq_id_max; ++s) {
                    if (seq_id[s]) {
                        ss << s%10;
                    } else {
                        ss << ".";
                    }
                }

                if (ubatch.token) {
                    LLM_LOG(true, "%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.token[i], vocab->token_to_piece(ubatch.token[i]).c_str(),
                            ubatch.pos[i], ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                } else {
                    LLM_LOG(true, "%s:  %4d: [embd], pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.pos[i], ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                }
            }
            LLM_LOG(true, "%s:   ]\n", __func__);
        }
    }
}

//
// interface implementation
//

struct llm_batch llm_batch_get_one(
               llm_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens =*/ n_tokens,
        /*tokens   =*/ tokens,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };
}

struct llm_batch llm_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llm_batch batch = {
        /*n_tokens =*/ 0,
        /*tokens   =*/ nullptr,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llm_token *) malloc(sizeof(llm_token) * n_tokens_alloc);
    }

    batch.pos      = (llm_pos *)     malloc(sizeof(llm_pos)      * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llm_seq_id **) malloc(sizeof(llm_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llm_seq_id *) malloc(sizeof(llm_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);

    return batch;
}

void llm_batch_free(struct llm_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}
