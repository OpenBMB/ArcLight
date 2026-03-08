/**
 * @file tokenizer.h
 * @brief vocabulary and tokenization for LLMs.
 * 
 * This module is transplanted from the vocab implementation of "llama.cpp".
 * The open-source LICENSE is decribed in the root directory of this project.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "cgraph.h"

#define LLM_TOKEN_NULL -1

typedef int32_t llm_token;

struct chat_msg {
    chat_msg(std::string r, std::string c)
        : role(std::move(r)), content(std::move(c)) {}
    chat_msg(std::string r, std::string c, std::string rc)
        : role(std::move(r)), content(std::move(c)), reasoning_content(std::move(rc)) {}

    std::string role;     // "system" | "user" | "assistant"
    std::string content;  // text
    std::string reasoning_content; // can be null；assistant use
};

enum llm_token_attr {
        LLM_TOKEN_ATTR_UNDEFINED    = 0,
        LLM_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLM_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLM_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLM_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLM_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLM_TOKEN_ATTR_BYTE         = 1 << 5,
        LLM_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLM_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLM_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLM_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

enum llm_token_type {
        LLM_TOKEN_TYPE_UNDEFINED    = 0,
        LLM_TOKEN_TYPE_NORMAL       = 1,
        LLM_TOKEN_TYPE_UNKNOWN      = 2,
        LLM_TOKEN_TYPE_CONTROL      = 3,
        LLM_TOKEN_TYPE_USER_DEFINED = 4,
        LLM_TOKEN_TYPE_UNUSED       = 5,
        LLM_TOKEN_TYPE_BYTE         = 6,
    };

enum llm_special_token {
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_EOT_ID,
    LLM_KV_TOKENIZER_EOM_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_CLS_ID,
    LLM_KV_TOKENIZER_MASK_ID,
    LLM_KV_TOKENIZER_FIM_PRE_ID,
    LLM_KV_TOKENIZER_FIM_SUF_ID,
    LLM_KV_TOKENIZER_FIM_MID_ID,
    LLM_KV_TOKENIZER_FIM_PAD_ID,
    LLM_KV_TOKENIZER_FIM_REP_ID,
    LLM_KV_TOKENIZER_FIM_SEP_ID,
    // may deprecated
    LLM_KV_TOKENIZER_PREFIX_ID,
    LLM_KV_TOKENIZER_SUFFIX_ID,
    LLM_KV_TOKENIZER_MIDDLE_ID,
};

struct token_data {
    std::string    text;
    float          score = 0.0f;
    llm_token_type type;
};

struct vocab_data {
    std::string                     model;
    std::string                     pre_model;
    std::vector<token_data>         vocab;
    std::vector<std::string>        merges;
    bool has_score = false;
    bool has_type  = false;
    // std::unordered_map<std::string, int> token_to_id;

    bool add_bos = false;
    bool read_add_bos = false;
    bool add_eos = false;
    bool read_add_eos = false;
    bool add_sep = false;
    bool read_add_sep = false;
    uint32_t bos_id = LLM_TOKEN_NULL;
    uint32_t eos_id = LLM_TOKEN_NULL;
    uint32_t pad_id = LLM_TOKEN_NULL;
    uint32_t eot_id = LLM_TOKEN_NULL;
    uint32_t eom_id = LLM_TOKEN_NULL;
    uint32_t unk_id = LLM_TOKEN_NULL;
    uint32_t sep_id = LLM_TOKEN_NULL;
    uint32_t mask_id = LLM_TOKEN_NULL;
    uint32_t fim_pre_id = LLM_TOKEN_NULL;
    uint32_t fim_suf_id = LLM_TOKEN_NULL;
    uint32_t fim_mid_id = LLM_TOKEN_NULL;
    uint32_t fim_pad_id = LLM_TOKEN_NULL;
    uint32_t fim_rep_id = LLM_TOKEN_NULL;

    bool add_space_prefix = false;
    bool read_add_space_prefix = false;
    bool remove_extra_whitespaces = false;
    bool read_remove_extra_whitespaces = false;
};

// pre-tokenization types
enum llm_tokenizer_pre_type {
    LLM_TOKENIZER_PRE_TYPE_DEFAULT        = 0,
    LLM_TOKENIZER_PRE_TYPE_LLAMA3         = 1,
    LLM_TOKENIZER_PRE_TYPE_DEEPSEEK_LLM   = 2,
    LLM_TOKENIZER_PRE_TYPE_DEEPSEEK_CODER = 3,
    LLM_TOKENIZER_PRE_TYPE_FALCON         = 4,
    LLM_TOKENIZER_PRE_TYPE_MPT            = 5,
    LLM_TOKENIZER_PRE_TYPE_STARCODER      = 6,
    LLM_TOKENIZER_PRE_TYPE_GPT2           = 7,
    LLM_TOKENIZER_PRE_TYPE_REFACT         = 8,
    LLM_TOKENIZER_PRE_TYPE_COMMAND_R      = 9,
    LLM_TOKENIZER_PRE_TYPE_STABLELM2      = 10,
    LLM_TOKENIZER_PRE_TYPE_QWEN2          = 11,
    LLM_TOKENIZER_PRE_TYPE_OLMO           = 12,
    LLM_TOKENIZER_PRE_TYPE_DBRX           = 13,
    LLM_TOKENIZER_PRE_TYPE_SMAUG          = 14,
    LLM_TOKENIZER_PRE_TYPE_PORO           = 15,
    LLM_TOKENIZER_PRE_TYPE_CHATGLM3       = 16,
    LLM_TOKENIZER_PRE_TYPE_CHATGLM4       = 17,
    LLM_TOKENIZER_PRE_TYPE_VIKING         = 18,
    LLM_TOKENIZER_PRE_TYPE_JAIS           = 19,
    LLM_TOKENIZER_PRE_TYPE_TEKKEN         = 20,
    LLM_TOKENIZER_PRE_TYPE_SMOLLM         = 21,
    LLM_TOKENIZER_PRE_TYPE_CODESHELL      = 22,
    LLM_TOKENIZER_PRE_TYPE_BLOOM          = 23,
    LLM_TOKENIZER_PRE_TYPE_GPT3_FINNISH   = 24,
    LLM_TOKENIZER_PRE_TYPE_EXAONE         = 25,
    LLM_TOKENIZER_PRE_TYPE_CHAMELEON      = 26,
    LLM_TOKENIZER_PRE_TYPE_MINERVA        = 27,
    LLM_TOKENIZER_PRE_TYPE_DEEPSEEK3_LLM  = 28,
    LLM_TOKENIZER_PRE_TYPE_GPT4O          = 29,
    LLM_TOKENIZER_PRE_TYPE_SUPERBPE       = 30,
    LLM_TOKENIZER_PRE_TYPE_TRILLION       = 31,
    LLM_TOKENIZER_PRE_TYPE_BAILINGMOE     = 32,
    LLM_TOKENIZER_PRE_TYPE_LLAMA4         = 33,
    LLM_TOKENIZER_PRE_TYPE_PIXTRAL        = 34,
    LLM_TOKENIZER_PRE_TYPE_SEED_CODER     = 35,
    LLM_TOKENIZER_PRE_TYPE_HUNYUAN        = 36,
    LLM_TOKENIZER_PRE_TYPE_KIMI_K2        = 37,
    LLM_TOKENIZER_PRE_TYPE_HUNYUAN_DENSE  = 38,
    LLM_TOKENIZER_PRE_TYPE_GROK_2         = 39,
};

enum llm_tokenizer_type {
        LLM_TOKENIZER_TYPE_NONE   = 0, // For models without vocab
        LLM_TOKENIZER_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLM_TOKENIZER_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLM_TOKENIZER_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLM_TOKENIZER_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLM_TOKENIZER_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLM_TOKENIZER_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

struct llm_vocab {
    struct token_data {
        std::string      text;
        float            score;
        llm_token_attr attr;
    };

    llm_vocab();
    ~llm_vocab();

    void load(vocab_data & vocab_data);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum llm_tokenizer_type     get_type()     const;
    enum llm_tokenizer_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (llm_token id) const;
    bool is_unknown     (llm_token id) const;
    bool is_control     (llm_token id) const;
    bool is_byte        (llm_token id) const;
    bool is_user_defined(llm_token id) const;
    bool is_unused      (llm_token id) const;
    bool is_eog         (llm_token id) const;

    uint8_t     token_to_byte(llm_token id) const;
    llm_token byte_to_token(uint8_t ch)     const;

    llm_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(llm_token id) const;

    const char *     token_get_text (llm_token id) const;
    float            token_get_score(llm_token id) const;
    llm_token_attr token_get_attr (llm_token id) const;

    llm_token token_bos() const;
    llm_token token_eos() const;
    llm_token token_eot() const;
    llm_token token_eom() const;
    llm_token token_unk() const;
    llm_token token_sep() const;
    llm_token token_nl () const;
    llm_token token_pad() const;
    llm_token token_mask() const;

    llm_token token_prefix() const;
    llm_token token_middle() const;
    llm_token token_suffix() const;

    llm_token token_fim_pre() const;
    llm_token token_fim_suf() const;
    llm_token token_fim_mid() const;
    llm_token token_fim_pad() const;
    llm_token token_fim_rep() const;
    llm_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llm_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<llm_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  llm_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(llm_token token) const;

    int32_t detokenize(
            const llm_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<llm_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

bool llm_vocab_is_eog(const struct llm_vocab * vocab, llm_token token);

std::vector<llm_token> common_tokenize(const struct llm_vocab * vocab, const std::string & text, bool add_special, bool parse_special);
std::string common_token_to_piece(const struct llm_vocab * vocab, llm_token token, bool special);
std::string common_detokenize(const struct llm_vocab * vocab, const std::vector<llm_token> & tokens, bool special);

void replace_all(std::string & s, const std::string & search, const std::string & replace);
std::string format(const char * fmt, ...);

struct unicode_cpt_flags;

uint32_t unicode_tolower(uint32_t cpt);
size_t unicode_len_utf8(char src);
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset);
std::string unicode_cpt_to_utf8(uint32_t cpt);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);
std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string & utf8);
unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt);
std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs);
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

struct unicode_cpt_flags {                                  // this is an endian-dependent struct
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // regex: \p{N}
        LETTER          = 0x0004,  // regex: \p{L}
        SEPARATOR       = 0x0008,  // regex: \p{Z}
        ACCENT_MARK     = 0x0010,  // regex: \p{M}
        PUNCTUATION     = 0x0020,  // regex: \p{P}
        SYMBOL          = 0x0040,  // regex: \p{S}
        CONTROL         = 0x0080,  // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
        WHITESPACE      = 0x0100,
        LOWERCASE       = 0x0200,
        UPPERCASE       = 0x0400,
        NFD             = 0x0800,
    };

    // codepoint type
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;  // regex: \p{N}
    uint16_t is_letter      : 1;  // regex: \p{L}
    uint16_t is_separator   : 1;  // regex: \p{Z}
    uint16_t is_accent_mark : 1;  // regex: \p{M}
    uint16_t is_punctuation : 1;  // regex: \p{P}
    uint16_t is_symbol      : 1;  // regex: \p{S}
    uint16_t is_control     : 1;  // regex: \p{C}
    // helper flags
    uint16_t is_whitespace  : 1;  // regex: \s
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    // decode from uint16
    inline unicode_cpt_flags(const uint16_t flags = 0) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        *reinterpret_cast<uint16_t*>(this) = flags;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        is_undefined   = (flags & UNDEFINED)   ? 1 : 0;
        is_number      = (flags & NUMBER)      ? 1 : 0;
        is_letter      = (flags & LETTER)      ? 1 : 0;
        is_separator   = (flags & SEPARATOR)   ? 1 : 0;
        is_accent_mark = (flags & ACCENT_MARK) ? 1 : 0;
        is_punctuation = (flags & PUNCTUATION) ? 1 : 0;
        is_symbol      = (flags & SYMBOL)      ? 1 : 0;
        is_control     = (flags & CONTROL)     ? 1 : 0;
        is_whitespace  = (flags & WHITESPACE)  ? 1 : 0;
        is_lowercase   = (flags & LOWERCASE)   ? 1 : 0;
        is_uppercase   = (flags & UPPERCASE)   ? 1 : 0;
        is_nfd         = (flags & NFD)         ? 1 : 0;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
    }

    inline uint16_t as_uint() const {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return *reinterpret_cast<const uint16_t*>(this);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        uint16_t result =
              is_undefined   * UNDEFINED
            + is_number      * NUMBER
            + is_letter      * LETTER
            + is_separator   * SEPARATOR
            + is_accent_mark * ACCENT_MARK
            + is_punctuation * PUNCTUATION
            + is_symbol      * SYMBOL
            + is_control     * CONTROL
            + is_whitespace  * WHITESPACE
            + is_lowercase   * LOWERCASE
            + is_uppercase   * UPPERCASE
            + is_nfd         * NFD
            ;

        return result;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
    }

    inline uint16_t category_flag() const {
        return this->as_uint() & MASK_CATEGORIES;
    }
};

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;
