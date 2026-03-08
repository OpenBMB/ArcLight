// ACKNOWLEDGEMENT: We use the open-sourced llama.cpp vocab implementation
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <forward_list>
#include <limits>
#include <climits>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <regex>
#include <codecvt>

#include "tokenizer.h"
#include "model.h"


// helpers
struct naive_trie {
    naive_trie() : has_value(false), value(0) {
    }
    void insert(const char * key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->has_value = true;
            this->value = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        } else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }
    std::pair<const char *, size_t> get_longest_prefix(const char * key, size_t len, size_t offset = 0) const {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        }

        return std::make_pair(key, offset);
    }
    const struct naive_trie * traverse(const char c) const {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        }

        return NULL;
    }
    std::map<char, struct naive_trie> children;
    bool has_value;
    llm_token value;
};


// tokenizers
struct llm_tokenizer {
    llm_tokenizer() {}
    virtual ~llm_tokenizer() = default;
};

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm : llm_tokenizer {
    llm_tokenizer_spm(const llm_vocab & /*vocab*/) {}
};

struct llm_tokenizer_spm_session {
    llm_tokenizer_spm_session(const llm_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llm_token> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (int i = 1; i < (int) symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLM_LOG(true, "left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llm_token> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.text_to_token(text);

        // Do we need to support is_unused?
        if (token != LLM_TOKEN_NULL) {
            output.push_back(token);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llm_token id = vocab.byte_to_token(symbol.text[j]);
                output.push_back(id);
            }
            return;
        }

        resegment(symbols[p->second.first], output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.text_to_token(text);

        if (token == LLM_TOKEN_NULL) {
            return;
        }

        if (static_cast<uint32_t>(token) >= vocab.n_tokens()) {
            return;
        }

        const auto & tok_data = vocab.get_token_data(token);

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llm_vocab & vocab;
    // currently unused
    // const llm_tokenizer_spm * spm_tokenizer;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
    std::map<std::string, std::pair<int, int>> rev_merge;
};

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llm_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() =  delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llm_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe : llm_tokenizer {
    llm_tokenizer_bpe(const llm_vocab & vocab) {
        assert(vocab.get_type() == LLM_TOKENIZER_TYPE_BPE);
        switch (vocab.get_pre_type()) {
            case LLM_TOKENIZER_PRE_TYPE_LLAMA3:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_DBRX:
            case LLM_TOKENIZER_PRE_TYPE_SMAUG:
                regex_exprs = {
                    // same as llama3
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_DEEPSEEK_LLM:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                    "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                    "\\s+$",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_DEEPSEEK3_LLM:
            case LLM_TOKENIZER_PRE_TYPE_HUNYUAN_DENSE:
                regex_exprs = {
                    "\\p{N}{1,3}",
                    "[一-龥぀-ゟ゠-ヿ]+",
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_DEEPSEEK_CODER:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?\\p{L}+",
                    "\\s?\\p{P}+",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_FALCON:
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "[0-9][0-9][0-9]",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_STARCODER:
            case LLM_TOKENIZER_PRE_TYPE_REFACT:
            case LLM_TOKENIZER_PRE_TYPE_COMMAND_R:
            case LLM_TOKENIZER_PRE_TYPE_SMOLLM:
            case LLM_TOKENIZER_PRE_TYPE_CODESHELL:
            case LLM_TOKENIZER_PRE_TYPE_EXAONE:
            case LLM_TOKENIZER_PRE_TYPE_MINERVA:
                regex_exprs = {
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_GPT2:
            case LLM_TOKENIZER_PRE_TYPE_MPT:
            case LLM_TOKENIZER_PRE_TYPE_OLMO:
            case LLM_TOKENIZER_PRE_TYPE_JAIS:
            case LLM_TOKENIZER_PRE_TYPE_TRILLION:
                regex_exprs = {
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_STABLELM2:
            case LLM_TOKENIZER_PRE_TYPE_QWEN2:
            case LLM_TOKENIZER_PRE_TYPE_HUNYUAN:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_PORO:
            case LLM_TOKENIZER_PRE_TYPE_BLOOM:
            case LLM_TOKENIZER_PRE_TYPE_GPT3_FINNISH:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_CHATGLM4:
                regex_exprs = {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_VIKING:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                    "\\p{N}",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_TEKKEN:
                // original regex from tokenizer.json
                // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                regex_exprs = {
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_CHAMELEON:
                // Note: in theory, the special token (sentinel and image token) regex_exprs below
                // are unnecessary, as they are split in `tokenizer_st_partition` anyway.
                regex_exprs = {
                    "<sentinel:[0-9]+>",  // Sentinel tokens
                    "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",  // Image tokens
                    "([\\t\\n]|    |  )",  // directly from tokenizer.json
                    "\\p{N}", // Individual digits
                    "[\\p{P}!-/:-@\\[-`{-~]",  // Punctuation, Isolated
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_GPT4O:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_KIMI_K2:
                regex_exprs = {
                    // K2 trigger pattern - this will activate the custom K2 handler in unicode.cpp
                    // The custom handler implements all K2 patterns with proper Han character exclusion
                    "\\p{Han}+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_SUPERBPE:
                regex_exprs = {
                    "\\p{N}+",
                    "(?=(\\d{3})+(?!\\d))",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_BAILINGMOE:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
                    // FIXME? Changed possessive quantifiers (?+ and ++) to greedy to avoid errors and imatrix hanging (tried atomic grouping but it's not supported?)
                    "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_SEED_CODER:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLM_TOKENIZER_PRE_TYPE_GROK_2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            default:
                // default regex for BPE tokenization pre-processing
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]",
                };
                break;
        }
    }

    std::vector<std::string> regex_exprs;
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const llm_vocab & vocab, const llm_tokenizer_bpe & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    static void append(const llm_token token_id, std::vector<llm_token> & output)  {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<llm_token> & output) const {
        if (vocab.get_add_bos()) {
            assert(vocab.token_bos() != LLM_TOKEN_NULL);
            output.push_back(vocab.token_bos());
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<llm_token> & output) const {
        if (vocab.get_add_eos()) {
            assert(vocab.token_eos() != LLM_TOKEN_NULL);
            output.push_back(vocab.token_eos());
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<llm_token> & output) const {
        if (vocab.get_add_bos() && output.size() >= 2 && output[1] == vocab.token_bos()) {
            LLM_LOG(true, 
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.get_add_eos() && output.size() >= 2 && *(output.end()-2) == vocab.token_eos()) {
            LLM_LOG(true, 
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    void tokenize(const std::string & text, std::vector<llm_token> & output) {
        int final_prev_index = -1;
        const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs);

        symbols_final.clear();

        for (const auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
            if (vocab.get_ignore_merges() && vocab.text_to_token(word) != LLM_TOKEN_NULL) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (int i = 1; i < (int) symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.text_to_token(str);

                if (token == LLM_TOKEN_NULL) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.text_to_token(byte_str);
                        if (token_multibyte != LLM_TOKEN_NULL) {
                            output.push_back(token_multibyte);
                        }
                    }
                } else {
                    output.push_back(token);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    const llm_vocab & vocab;
    const llm_tokenizer_bpe & tokenizer;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};


// WPM tokenizer
struct llm_tokenizer_wpm : llm_tokenizer {
    llm_tokenizer_wpm(const llm_vocab & /*vocab*/) {}
};

struct llm_tokenizer_wpm_session {
    llm_tokenizer_wpm_session(const llm_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llm_token> & output) {
        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);
        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string & word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len() + 1); j > i; j--) {
                    auto id = vocab.text_to_token(word1.substr(i, j - i));
                    if (id != LLM_TOKEN_NULL) {
                        output.push_back(id);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.token_unk());
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    static std::vector<std::string> preprocess(const std::string & text)  {
        const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts_nfd) {
            const auto flags = unicode_cpt_flags_from_cpt(cpt);

            if (flags.is_whitespace) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            assert (!flags.is_separator);
            if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (flags.is_punctuation || ( cpt < 0x7F && flags.is_symbol ) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }

    static bool is_chinese_char(uint32_t cpt) {
        return
            (cpt >= 0x04E00 && cpt <= 0x09FFF) ||
            (cpt >= 0x03400 && cpt <= 0x04DBF) ||
            (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
            (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
            (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
            (cpt >= 0x2B920 && cpt <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (cpt >= 0x0F900 && cpt <= 0x0FAFF) ||
            (cpt >= 0x2F800 && cpt <= 0x2FA1F);
            //(cpt >= 0x3000  && cpt <= 0x303F)  ||
            //(cpt >= 0xFF00  && cpt <= 0xFFEF);
    }

private:
    const llm_vocab & vocab;
    // currently unused
    // const llm_tokenizer_wpm * wpm_tokenizer;
};


// UGM tokenizer
struct llm_tokenizer_ugm : llm_tokenizer {
    llm_tokenizer_ugm(const llm_vocab & vocab, const std::vector<char> & precompiled_charsmap) {
        if (precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t *) &precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);
            if (xcda_blob_size + charsmap_offset >= precompiled_charsmap.size()) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }

            // Next xcda_blob_size bytes contain entries of XOR-compressed compact
            // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
            xcda_array = (const uint32_t *) &precompiled_charsmap[charsmap_offset];
            xcda_array_size = xcda_blob_size / sizeof(uint32_t);
            charsmap_offset += xcda_blob_size;

            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = precompiled_charsmap.size() - charsmap_offset;
        }

        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            const auto & token_data = vocab.get_token_data(id);

            if (vocab.is_normal(id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (vocab.is_normal(id) ||
                vocab.is_user_defined(id) ||
                vocab.is_unused(id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (vocab.is_user_defined(id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }

    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";

    const char * prefix_replacements = NULL;
    size_t prefix_replacements_size = 0;

    const uint32_t * xcda_array = NULL;
    size_t xcda_array_size = 0;

    struct naive_trie user_defined_token_matcher;

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;

    float unknown_token_score_penalty = 10.0;
    float unknown_token_score;

    struct naive_trie token_matcher;
};

struct llm_tokenizer_ugm_session {
    llm_tokenizer_ugm_session(const llm_vocab & vocab, const llm_tokenizer_ugm & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
    void tokenize(const std::string & text, std::vector<llm_token> & output) {
        // get current size of output (for reversal later)
        size_t output_size = output.size();

        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }

        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.token_unk(), 0, -DBL_MAX});
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.token_unk(), 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization & current_best = tokenization_results[input_offset];
            const struct naive_trie * node = tokenizer.token_matcher.traverse(normalized[prefix_offset++]);

            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    llm_token token_id = node->value;
                    const auto & token_data = vocab.get_token_data(token_id);

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = vocab.is_user_defined(token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if (!single_codepoint_token_found) {
                const double challenger_score = current_best.score_sum + tokenizer.unknown_token_score;
                prefix_offset = input_offset + n_utf8_code_units;
                struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    struct best_tokenization challenger = { vocab.token_unk(), input_offset, challenger_score };
                    current_champ = challenger;
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        bool is_prev_unknown = false;
        for (struct best_tokenization & tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
            bool is_unknown = tokenization.token_id == vocab.token_unk();
            if (!(is_prev_unknown && is_unknown)) {
                output.push_back(tokenization.token_id);
            }
            if (tokenization.input_offset == 0) {
                break;
            }
            is_prev_unknown = is_unknown;
        }

        // reverse the output since we added tokens starting from the end of the input
        std::reverse(output.begin() + output_size, output.end());
    }

private:

    // helper structure for returning normalization results
    struct normalization_result {
        const char * normalized;
        size_t normalized_len;
        size_t consumed_input;
    };

    void normalize(const std::string& input, std::string * normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.get_escape_whitespaces() ? tokenizer.escaped_space : " ";

        const bool shall_prepend_space = !vocab.get_treat_whitespace_as_suffix() && vocab.get_add_space_prefix();
        const bool shall_append_space  =  vocab.get_treat_whitespace_as_suffix() && vocab.get_add_space_prefix();
        const bool shall_merge_spaces  =  vocab.get_remove_extra_whitespaces();

        bool is_space_prepended = false;
        bool processing_non_ws = false;

        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                } else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }

            input_offset += norm_res.consumed_input;
        }

        if (shall_append_space) {
            normalized->append(space);
        }
    }

    /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Each bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
    struct xcda_array_view {
    public:
        xcda_array_view(const uint32_t * xcda_array, size_t xcda_array_size) : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {
        }
        uint32_t get_base(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
        }
        uint32_t get_lcheck(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) | 0xff);
        }
        bool get_leaf(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 8) & 1;
        }
        uint32_t get_value(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) - 1);
        }
    private:
        uint32_t get_node(size_t index) {
            if (index > xcda_array_size) {
                throw std::runtime_error("Index out of array bounds in XCDA array!");
            }
            return xcda_array[index];
        }
        const uint32_t * xcda_array;
        size_t xcda_array_size;
    };

    // this structure stores the best tokenization so far at input_offset
    struct best_tokenization {
        llm_token token_id;
        size_t input_offset;
        double score_sum;
    };

    struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match =
           tokenizer.user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }

        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (tokenizer.xcda_array_size > 0) {
            struct xcda_array_view xcda_view(tokenizer.xcda_array, tokenizer.xcda_array_size);

            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;
            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);
            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
        }

        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= tokenizer.prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char * prefix_replacement = &(tokenizer.prefix_replacements)[longest_prefix_offset];
            return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
        }

        // check if the input prefix contains a valid sequence of UTF-8 code units
        try {
            // if yes, return this sequence unmodified
            size_t prefix_offset = input_offset;
            unicode_cpt_from_utf8(input, prefix_offset);
            return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
        } catch (std::invalid_argument & /*ex*/) {
            // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
            return { "\xEF\xBF\xBD", 3, 1 };
        }
    }

    const llm_vocab & vocab;
    const llm_tokenizer_ugm & tokenizer;
};


// RWKV tokenizer
static std::vector<uint8_t> llm_unescape_rwkv_token(const std::string & escaped) {
    std::vector<uint8_t> output;
    output.reserve(escaped.size());

    // Parser state
    bool escaping = false;
    uint8_t hex_remaining = 0;
    uint8_t hex_acc = 0;

    // Step through characters, performing parsing
    for (const char & c : escaped) {
        // If we're parsing a hex code, interpret the next character
        if (hex_remaining != 0) {
            uint8_t value = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            hex_acc = (hex_acc << 4) + value;

            hex_remaining -= 1;
            if (hex_remaining == 0) {
                output.push_back(hex_acc);
                hex_acc = 0;
            }

            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            if (c == 't') {
                output.push_back('\t');
            } else if (c == 'n') {
                output.push_back('\n');
            } else if (c == 'r') {
                output.push_back('\r');
            } else if (c == 'x') {
                hex_remaining = 2;
            } else {
                output.push_back(c);
            }

            escaping = false;
            continue;
        }

        if (c == '\\') {
            escaping = true;
            continue;
        }

        output.push_back(c);
    }

    return output;
}

struct llm_tokenizer_rwkv : llm_tokenizer {
    llm_tokenizer_rwkv(const llm_vocab & vocab) {
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.

        // build trie
        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            const auto & data = vocab.get_token_data(id);
            const auto text = llm_unescape_rwkv_token(data.text);
            token_matcher.insert((const char *) text.data(), text.size(), id);
        }
    }

    struct naive_trie token_matcher;
};

struct llm_tokenizer_rwkv_session {
    llm_tokenizer_rwkv_session(const llm_vocab & vocab, const llm_tokenizer_rwkv & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<llm_token> & output) {
        uint32_t position = 0;
        while (position < text.size()) {
            const struct naive_trie * node = tokenizer.token_matcher.traverse(text[position]);
            if (node == NULL) {
                // no matching token found, add unknown token
                output.push_back(vocab.token_unk());
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            uint32_t token_id = 0;
            uint32_t token_length = 0;
            while (node != NULL) {
                if (node->has_value) {
                    token_id = node->value;
                    token_length = position + 1;
                }
                node = node->traverse(text[++position]);
            }

            // add the longest matching token
            output.push_back(token_id);
            position = token_length;
        }
    }

private:
    const llm_vocab & vocab;
    const llm_tokenizer_rwkv & tokenizer;
};

struct llm_tokenizer_plamo2 : llm_tokenizer {
    llm_tokenizer_plamo2(const llm_vocab & vocab) {
        build(vocab);
    }

    void build(const llm_vocab & vocab) {
        // Reset internal structures
        tokens_.clear();
        bytes_.assign(256, 0);
        to_suffix_id_.clear();
        table_.clear();

        // Build token list and byte mapping
        std::unordered_map<std::string, float> suffix_to_score;
        std::unordered_map<std::string, llm_token> token_to_id;

        for (size_t token_id = 0; token_id < vocab.n_tokens(); ++token_id) {
            const auto & entry = vocab.get_token_data(token_id);
            tokens_.push_back(entry.text);
            token_to_id[entry.text] = static_cast<llm_token>(token_id);

            // Handle byte tokens
            if (vocab.is_byte(token_id)) {
                if (entry.text.length() == 6 && entry.text.substr(0, 3) == "<0x" && entry.text.back() == '>') {
                    std::string hex_str = entry.text.substr(3, 2);
                    int byte_val = std::stoi(hex_str, nullptr, 16);
                    bytes_[byte_val] = static_cast<llm_token>(token_id);
                }
                continue;
            }

            // Add token and all its suffixes to suffix_to_score
            suffix_to_score[entry.text] = entry.score;

            // Extract suffixes character by character (UTF-8 aware)
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(entry.text);
            for (size_t i = 1; i < cpts.size(); ++i) {
                std::string suffix;
                for (size_t j = i; j < cpts.size(); ++j) {
                    suffix += unicode_cpt_to_utf8(cpts[j]);
                }
                if (suffix_to_score.find(suffix) == suffix_to_score.end()) {
                    suffix_to_score[suffix] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        // Check that all byte tokens are set
        for (int i = 0; i < 256; ++i) {
            if (bytes_[i] == 0) {
                throw std::runtime_error("Byte token for <0x" + std::to_string(i) + "> is not set");
            }
        }

        // Build suffix list in lexicographical order of reversed strings
        std::vector<std::string> suffixes;
        for (const auto & pair : suffix_to_score) {
            suffixes.push_back(pair.first);
        }
        suffixes.push_back("");  // Empty suffix

        std::sort(suffixes.begin(), suffixes.end(), [](const std::string & a, const std::string & b) {
            std::string rev_a(a.rbegin(), a.rend());
            std::string rev_b(b.rbegin(), b.rend());
            return rev_a < rev_b;
        });

        // Build suffix_to_id and to_suffix_id_
        std::unordered_map<std::string, int32_t> suffix_to_id;
        int32_t num_pieces = 0;

        for (const auto & suffix : suffixes) {
            suffix_to_id[suffix] = num_pieces;
            if (!suffix.empty()) {
                std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);

                std::string remaining;
                for (size_t i = 1; i < cpts.size(); ++i) {
                    remaining += unicode_cpt_to_utf8(cpts[i]);
                }

                int64_t piece_code = (static_cast<int64_t>(cpts[0]) << 32) | suffix_to_id[remaining];
                to_suffix_id_[piece_code] = num_pieces;

                // Count number of pieces for this suffix
                int32_t pieces_for_suffix = 1; // sentinel row
                for (int32_t piece_length = static_cast<int32_t>(cpts.size()); piece_length > 0; --piece_length) {
                    std::string piece;
                    for (int32_t i = 0; i < piece_length; ++i) {
                        piece += unicode_cpt_to_utf8(cpts[i]);
                    }
                    if (suffix_to_score.find(piece) != suffix_to_score.end()) {
                        pieces_for_suffix++;
                    }
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces++;  // Empty suffix contributes one piece (sentinel row)
            }
        }

        // Build flattened table
        table_.resize(num_pieces, std::vector<int32_t>(4, 0));
        int32_t table_idx = 0;

        for (const auto & suffix : suffixes) {
            // Add all prefixes of the suffix to the table (in decreasing order of length)
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);
            for (int32_t piece_length = static_cast<int32_t>(cpts.size()); piece_length > 0; --piece_length) {
                std::string piece;
                for (int32_t i = 0; i < piece_length; ++i) {
                    piece += unicode_cpt_to_utf8(cpts[i]);
                }

                auto score_it = suffix_to_score.find(piece);
                if (score_it == suffix_to_score.end()) {
                    continue;
                }

                table_[table_idx][TABLE_PIECE_LENGTH] = piece_length;
                auto token_it = token_to_id.find(piece);
                table_[table_idx][TABLE_TOKEN_ID] = (token_it != token_to_id.end()) ? token_it->second : -1;

                float score = score_it->second;
                table_[table_idx][TABLE_SCORE] = std::isfinite(score) ?
                    static_cast<int32_t>(std::round(score * 1e4)) : INVALID_SCORE;
                table_[table_idx][TABLE_PIECE_ID] = suffix_to_id[piece];

                table_idx++;
            }

            // Add sentinel row
            table_[table_idx][TABLE_PIECE_LENGTH] = 1;
            table_[table_idx][TABLE_TOKEN_ID] = -1;
            table_[table_idx][TABLE_SCORE] = UNKNOWN_SCORE;
            table_idx++;
        }
    }

    std::vector<llm_token> encode(const std::string & text) const {
        std::vector<uint32_t> unicode_data = unicode_cpts_from_utf8(text);
        // Skip the first code point if it is a BOM (Byte Order Mark)
        if (!unicode_data.empty() && unicode_data[0] == 0xFEFF) {
            unicode_data.erase(unicode_data.begin());
        }

        if (unicode_data.empty()) {
            return {};
        }

        const size_t data_len = unicode_data.size();

        // Initialize scores array (dynamic programming)
        std::vector<int64_t> scores(data_len + 1, static_cast<int64_t>(1) << 60);
        scores[data_len] = 0;

        // Path array to track best tokenization
        std::vector<std::vector<int32_t>> path(data_len + 1, std::vector<int32_t>(3, 0));

        int32_t suffix_id = 0;

        // Process from end to beginning
        for (int i = static_cast<int>(data_len) - 1; i >= 0; --i) {
            uint32_t c = unicode_data[i];

            // Find next suffix ID
            for (size_t p = suffix_id; p < table_.size(); ++p) {
                int64_t piece_code = (static_cast<int64_t>(c) << 32) | table_[p][TABLE_PIECE_ID];
                auto it = to_suffix_id_.find(piece_code);
                suffix_id = (it != to_suffix_id_.end()) ? it->second : 0;

                if (suffix_id > 0 || table_[p][TABLE_SCORE] == UNKNOWN_SCORE) {
                    break;
                }
            }

            // Update best path
            for (size_t p = suffix_id; p < table_.size(); ++p) {
                int32_t score = table_[p][TABLE_SCORE];
                if (score > INVALID_SCORE) {
                    int32_t piece_length = table_[p][TABLE_PIECE_LENGTH];
                    int64_t s = scores[i + piece_length] - score;

                    if (s < scores[i]) {
                        scores[i] = s;
                        path[i][PATH_TOKEN_LENGTH] = piece_length;
                        path[i][PATH_TOKEN_ID] = table_[p][TABLE_TOKEN_ID];
                        path[i][PATH_NUM_TOKENS] = path[i + piece_length][PATH_NUM_TOKENS] + 1;

                        if (score == UNKNOWN_SCORE) {
                            // Add UTF-8 byte count
                            path[i][PATH_NUM_TOKENS] += (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
                        }
                    }
                }

                if (score == UNKNOWN_SCORE) {
                    break;
                }
            }
        }

        // Decode the best path
        std::vector<llm_token> token_ids;
        token_ids.reserve(path[0][PATH_NUM_TOKENS]);

        int pos = 0;
        while (pos < static_cast<int>(data_len)) {
            if (path[pos][PATH_TOKEN_ID] >= 0) {
                token_ids.push_back(path[pos][PATH_TOKEN_ID]);
            } else {
                // Fall back to byte tokens
                uint32_t c = unicode_data[pos];
                int s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);

                for (int i = 0; i < s; ++i) {
                    uint8_t b;
                    if (s == 1) {
                        b = c;
                    } else {
                        if (i == 0) {
                            b = (0xF00 >> s) & 0xFF;
                        } else {
                            b = 0x80;
                        }
                    }
                    token_ids.push_back(bytes_[b | ((c >> ((s - i - 1) * 6)) & 0x3F)]);
                }
            }

            assert(path[pos][PATH_TOKEN_LENGTH] > 0);
            pos += path[pos][PATH_TOKEN_LENGTH];
        }

        return token_ids;
    }
private:
    // Constants for table structure
    static constexpr int32_t TABLE_PIECE_LENGTH = 0;
    static constexpr int32_t TABLE_TOKEN_ID     = 1;
    static constexpr int32_t TABLE_SCORE        = 2;
    static constexpr int32_t TABLE_PIECE_ID     = 3;

    // Constants for path array
    static constexpr int32_t PATH_TOKEN_LENGTH  = 0;
    static constexpr int32_t PATH_TOKEN_ID      = 1;
    static constexpr int32_t PATH_NUM_TOKENS    = 2;

    // Score constants
    static constexpr int32_t INVALID_SCORE = -20000000;
    static constexpr int32_t UNKNOWN_SCORE = -10000000;

    // List of tokens in the vocabulary
    std::vector<std::string> tokens_;

    // Mapping from byte code point to token ID (for byte fallback)
    std::vector<llm_token> bytes_;

    // Mapping from piece code to suffix ID
    std::unordered_map<int64_t, int32_t> to_suffix_id_;

    // Flattened table representing the Trie structure
    // Each row contains: [piece_length, token_id, score, piece_id]
    std::vector<std::vector<int32_t>> table_;
};

struct llm_tokenizer_plamo2_session {
    llm_tokenizer_plamo2_session(const llm_tokenizer_plamo2 & tokenizer) : tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<llm_token> & output) {
        std::vector<llm_token> tokens = tokenizer.encode(text);
        output.insert(output.end(), tokens.begin(), tokens.end());
    }

private:
    const llm_tokenizer_plamo2 & tokenizer;
};


// impl
typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llm_token _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llm_token) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            assert(_offset >= 0);
            assert(_length >= 1);
            assert(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llm_token token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

struct llm_vocab::impl {
    uint32_t n_token_types = 0; // for BERT-style token types

    std::string tokenizer_model;
    std::string tokenizer_pre;

    enum llm_tokenizer_type     type     = LLM_TOKENIZER_TYPE_SPM;
    enum llm_tokenizer_pre_type pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    // default llm special tokens
    // TODO: should we set all of these to LLM_TOKEN_NULL?
    llm_token special_bos_id  = 1;
    llm_token special_eos_id  = 2;
    llm_token special_eot_id  = LLM_TOKEN_NULL;
    llm_token special_eom_id  = LLM_TOKEN_NULL;
    llm_token special_unk_id  = 0;
    llm_token special_sep_id  = LLM_TOKEN_NULL;
    llm_token special_pad_id  = LLM_TOKEN_NULL;
    llm_token special_mask_id = LLM_TOKEN_NULL;

    llm_token linefeed_id = 13;

    // fim tokens
    llm_token special_fim_pre_id = LLM_TOKEN_NULL;
    llm_token special_fim_suf_id = LLM_TOKEN_NULL;
    llm_token special_fim_mid_id = LLM_TOKEN_NULL;
    llm_token special_fim_pad_id = LLM_TOKEN_NULL;
    llm_token special_fim_rep_id = LLM_TOKEN_NULL; // repo
    llm_token special_fim_sep_id = LLM_TOKEN_NULL; // file separator

    // tokenizer flags
    bool add_space_prefix           = false;
    bool add_bos                    = false;
    bool add_eos                    = false;
    bool add_sep                    = false;
    bool ignore_merges              = false;
    bool clean_spaces               = false;  // clean_up_tokenization_spaces
    bool remove_extra_whitespaces   = false;
    bool escape_whitespaces         = true;
    bool treat_whitespace_as_suffix = false;

    std::unordered_map<std::string, llm_token> token_to_id;
    std::vector<token_data>                      id_to_token;

    std::vector<llm_token> cache_special_tokens;
    std::vector<std::string> cache_token_to_piece; // llm_token_to_piece(special = true);
    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string> & p) const {
            return std::hash<std::string>{}(p.first) ^  //create some hash for pair
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;

    // set of all tokens that cause "end of generation"
    std::set<llm_token> special_eog_ids;

    std::unique_ptr<llm_tokenizer> tokenizer;

    std::vector<char> precompiled_charsmap;

    impl(const llm_vocab & vocab) : vocab(vocab) {
    }

    ~impl() = default;

    void load(vocab_data & vocab_data);

    enum llm_tokenizer_type get_type() const;

    std::string type_name() const;

    bool is_normal      (llm_token id) const;
    bool is_unknown     (llm_token id) const;
    bool is_control     (llm_token id) const;
    bool is_byte        (llm_token id) const;
    bool is_user_defined(llm_token id) const;
    bool is_unused      (llm_token id) const;
    bool is_eog         (llm_token id) const;

    uint8_t token_to_byte(llm_token id) const;

    llm_token_attr token_get_attr(llm_token id) const;

    void init_tokenizer(enum llm_tokenizer_type type);

    void tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const;

    std::string token_to_piece_for_cache(
                  llm_token   token,
                         bool   special) const;


    std::vector<llm_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llm_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

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
    const llm_vocab & vocab;
};

void llm_vocab::impl::load(vocab_data & vocab_data) {
    // determine vocab type
    {
        tokenizer_model = vocab_data.model;
        assert(tokenizer_model == "gpt2");
        tokenizer_pre = vocab_data.pre_model;
        // n_token_types = vocab_data.vocab.size();

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
            type = LLM_TOKENIZER_TYPE_NONE;
            // default special tokens
            special_bos_id  = LLM_TOKEN_NULL;
            special_eos_id  = LLM_TOKEN_NULL;
            special_unk_id  = LLM_TOKEN_NULL;
            special_sep_id  = LLM_TOKEN_NULL;
            special_pad_id  = LLM_TOKEN_NULL;
            special_mask_id = LLM_TOKEN_NULL;
            linefeed_id     = LLM_TOKEN_NULL;
            // read vocab size from metadata
            uint32_t n_tokens = vocab_data.vocab.size();
            if (n_tokens) {
                LLM_LOG(true, "%s: adding %u dummy tokens\n", __func__, n_tokens);
                id_to_token.resize(n_tokens);
            }

            return;
        }

        if (tokenizer_model == "llama") {
            type = LLM_TOKENIZER_TYPE_SPM;

            // default special tokens
            special_bos_id  = 1;
            special_eos_id  = 2;
            special_unk_id  = 0;
            special_sep_id  = LLM_TOKEN_NULL;
            special_pad_id  = LLM_TOKEN_NULL;
            special_mask_id = LLM_TOKEN_NULL;
        } else if (tokenizer_model == "bert") {
            type = LLM_TOKENIZER_TYPE_WPM;

            // default special tokens
            special_bos_id  = 101;
            special_eos_id  = LLM_TOKEN_NULL;
            special_unk_id  = 100;
            special_sep_id  = 102;
            special_pad_id  = 0;
            special_mask_id = 103;

            add_sep = true;
        } else if (tokenizer_model == "gpt2") {
            type = LLM_TOKENIZER_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int n_merges = vocab_data.merges.size();
            for (int i = 0; i < n_merges; i++) {
                const std::string word = vocab_data.merges[i];
                //assert(unicode_cpts_from_utf8(word).size() > 0);
                std::string first;
                std::string second;
                const size_t pos = word.find(' ', 1);
                if (pos != std::string::npos) {
                    first  = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }
                bpe_ranks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            special_bos_id  = 11;
            special_eos_id  = 11;
            special_unk_id  = LLM_TOKEN_NULL;
            special_sep_id  = LLM_TOKEN_NULL;
            special_pad_id  = LLM_TOKEN_NULL;
            special_mask_id = LLM_TOKEN_NULL;
        } else if (tokenizer_model == "rwkv") {
            type = LLM_TOKENIZER_TYPE_RWKV;

            // default special tokens
            special_bos_id = LLM_TOKEN_NULL;
            special_eos_id = LLM_TOKEN_NULL;
            special_unk_id = LLM_TOKEN_NULL;
            special_sep_id = LLM_TOKEN_NULL;
            special_pad_id = LLM_TOKEN_NULL;
        } else if (tokenizer_model == "plamo2") {
            type = LLM_TOKENIZER_TYPE_PLAMO2;

            // PLaMo-2 default special tokens (these will be overridden by model config)
            special_bos_id = 1;  // <|plamo:bos|>
            special_eos_id = 2;  // <|plamo:eos|>
            special_unk_id = 0;  // <|plamo:unk|>
            special_sep_id = LLM_TOKEN_NULL;
            special_pad_id = 3;  // <|plamo:pad|>
            special_mask_id = LLM_TOKEN_NULL;
        } else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (type == LLM_TOKENIZER_TYPE_BPE) {
            add_space_prefix = false;
            clean_spaces = true;
            if (tokenizer_pre.empty()) {
                LLM_LOG(true, "%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                LLM_LOG(true, "%s:                                             \n", __func__);
                LLM_LOG(true, "%s: ************************************        \n", __func__);
                LLM_LOG(true, "%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                LLM_LOG(true, "%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                LLM_LOG(true, "%s: ************************************        \n", __func__);
                LLM_LOG(true, "%s:                                             \n", __func__);
                pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3"  ||
                    tokenizer_pre == "falcon-h1" ||
                    tokenizer_pre == "pixtral"  ||
                    tokenizer_pre == "midm-2.0" ||
                    tokenizer_pre == "lfm2") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_LLAMA3;
                ignore_merges = true;
                add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_DEEPSEEK_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_DEEPSEEK_CODER;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-v3") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_DEEPSEEK3_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "falcon") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "gigachat"   ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "a.x-4.0" ||
                    tokenizer_pre == "mellum") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "jina-v1-en" ||
                    tokenizer_pre == "jina-v2-code" ||
                    tokenizer_pre == "roberta-bpe") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_GPT2;
                add_sep = true;
            } else if (
                    tokenizer_pre == "refact") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_COMMAND_R;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "qwen2" ||
                    tokenizer_pre == "deepseek-r1-qwen") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_QWEN2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_PORO;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "glm4" ||
                tokenizer_pre == "chatglm-bpe") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_CHATGLM4;
                special_bos_id = LLM_TOKEN_NULL;
            } else if (
                tokenizer_pre == "viking") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_VIKING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_TEKKEN;
                clean_spaces = false;
                ignore_merges = true;
                add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_SMOLLM;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_EXAONE;
            } else if (
                tokenizer_pre == "exaone4") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_GPT2;
            } else if (
                tokenizer_pre == "chameleon") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_CHAMELEON;
                add_bos = true;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minerva-7b") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_MINERVA;
            } else if (
                tokenizer_pre == "megrez") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_QWEN2;
            } else if (
                    tokenizer_pre == "gpt-4o" ||
                    tokenizer_pre == "llama4") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_GPT4O;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "superbpe") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_SUPERBPE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "trillion") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_TRILLION;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "bailingmoe" ||
                tokenizer_pre == "llada-moe") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_BAILINGMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "seed-coder") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_SEED_CODER;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_HUNYUAN;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan-dense") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_HUNYUAN_DENSE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "kimi-k2") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_KIMI_K2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "grok-2") {
                pre_type = LLM_TOKENIZER_PRE_TYPE_GROK_2;
                clean_spaces = false;
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
        } else if (type == LLM_TOKENIZER_TYPE_SPM) {
            pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
            add_space_prefix = true;
            clean_spaces = false;
            add_bos = true;
            add_eos = false;
        } else if (type == LLM_TOKENIZER_TYPE_WPM) {
            pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = true;
            add_bos = true;
            add_eos = false;
            add_sep = true;
        } else if (type == LLM_TOKENIZER_TYPE_UGM) {
            pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
            add_bos = false;
            add_eos = true;
        } else if (type == LLM_TOKENIZER_TYPE_RWKV) {
            pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = false;
            add_bos = false;
            add_eos = false;
        } else {
            pre_type = LLM_TOKENIZER_PRE_TYPE_DEFAULT;
        }

        if (vocab_data.read_add_space_prefix) add_space_prefix = vocab_data.add_space_prefix;
        if (vocab_data.read_remove_extra_whitespaces) remove_extra_whitespaces = vocab_data.remove_extra_whitespaces;
    }

    uint32_t n_tokens = vocab_data.vocab.size();
    id_to_token.resize(n_tokens);

    for (uint32_t i = 0; i < n_tokens; i++) {
        std::string word = vocab_data.vocab[i].text;
        if (word.empty()) {
            LLM_LOG(true, "%s: empty token at index %u\n", __func__, i);
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        token_to_id[word] = i;
        max_token_len = std::max(max_token_len, (int) word.size());

        auto & token_data = id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = vocab_data.vocab[i].score;
        token_data.attr  = LLM_TOKEN_ATTR_NORMAL;

        if (vocab_data.has_type) {  //TODO: remove, required until per token attributes are available from NNUF file
            switch(vocab_data.vocab[i].type) {
                case LLM_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLM_TOKEN_ATTR_UNKNOWN;      break;
                case LLM_TOKEN_TYPE_UNUSED:       token_data.attr = LLM_TOKEN_ATTR_UNUSED;       break;
                case LLM_TOKEN_TYPE_NORMAL:       token_data.attr = LLM_TOKEN_ATTR_NORMAL;       break;
                case LLM_TOKEN_TYPE_CONTROL:      token_data.attr = LLM_TOKEN_ATTR_CONTROL;      break;
                case LLM_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLM_TOKEN_ATTR_USER_DEFINED; break;
                case LLM_TOKEN_TYPE_BYTE:         token_data.attr = LLM_TOKEN_ATTR_BYTE;         break;
                case LLM_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLM_TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = LLM_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    assert(id_to_token.size() == token_to_id.size());

    init_tokenizer(type);

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (type == LLM_TOKENIZER_TYPE_SPM) {
        try {
            linefeed_id = vocab.byte_to_token('\n');
        } catch (const std::exception & e) {
            LLM_LOG(true, "%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            linefeed_id = special_pad_id;
        }
    } else if (type == LLM_TOKENIZER_TYPE_WPM) {
        linefeed_id = special_pad_id;
    } else if (type == LLM_TOKENIZER_TYPE_RWKV) {
        const std::vector<int> ids = tokenize("\n", false);
        assert(!ids.empty() && "model vocab missing newline token");
        linefeed_id = ids[0];
    } else {
        const std::vector<int> ids = tokenize("\n", false);

        //assert(!ids.empty() && "model vocab missing newline token");
        if (ids.empty()) {
            LLM_LOG(true, "%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            linefeed_id = special_pad_id;
        } else {
            linefeed_id = ids[0];
        }
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_special_token, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     special_bos_id     },
            { LLM_KV_TOKENIZER_EOS_ID,     special_eos_id     },
            { LLM_KV_TOKENIZER_EOT_ID,     special_eot_id     },
            { LLM_KV_TOKENIZER_EOM_ID,     special_eom_id     },
            { LLM_KV_TOKENIZER_UNK_ID,     special_unk_id     },
            { LLM_KV_TOKENIZER_SEP_ID,     special_sep_id     },
            { LLM_KV_TOKENIZER_PAD_ID,     special_pad_id     },
            { LLM_KV_TOKENIZER_MASK_ID,    special_mask_id    },
            { LLM_KV_TOKENIZER_FIM_PRE_ID, special_fim_pre_id },
            { LLM_KV_TOKENIZER_FIM_SUF_ID, special_fim_suf_id },
            { LLM_KV_TOKENIZER_FIM_MID_ID, special_fim_mid_id },
            { LLM_KV_TOKENIZER_FIM_PAD_ID, special_fim_pad_id },
            { LLM_KV_TOKENIZER_FIM_REP_ID, special_fim_rep_id },
            { LLM_KV_TOKENIZER_FIM_SEP_ID, special_fim_sep_id },

            // deprecated
            { LLM_KV_TOKENIZER_PREFIX_ID, special_fim_pre_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, special_fim_suf_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, special_fim_mid_id },
        };

        if (vocab_data.bos_id != LLM_TOKEN_NULL) special_bos_id = vocab_data.bos_id;
        if (vocab_data.eos_id != LLM_TOKEN_NULL) special_eos_id = vocab_data.eos_id;
        if (vocab_data.unk_id != LLM_TOKEN_NULL) special_unk_id = vocab_data.unk_id;
        if (vocab_data.sep_id != LLM_TOKEN_NULL) special_sep_id = vocab_data.sep_id;
        if (vocab_data.pad_id != LLM_TOKEN_NULL) special_pad_id = vocab_data.pad_id;
        if (vocab_data.mask_id != LLM_TOKEN_NULL) special_mask_id = vocab_data.mask_id;
        if (vocab_data.eot_id != LLM_TOKEN_NULL) special_eot_id = vocab_data.eot_id;
        if (vocab_data.eom_id != LLM_TOKEN_NULL) special_eom_id = vocab_data.eom_id;
        if (vocab_data.fim_pre_id != LLM_TOKEN_NULL) special_fim_pre_id = vocab_data.fim_pre_id;
        if (vocab_data.fim_suf_id != LLM_TOKEN_NULL) special_fim_suf_id = vocab_data.fim_suf_id;
        if (vocab_data.fim_mid_id != LLM_TOKEN_NULL) special_fim_mid_id = vocab_data.fim_mid_id;
        if (vocab_data.fim_pad_id != LLM_TOKEN_NULL) special_fim_pad_id = vocab_data.fim_pad_id;
        if (vocab_data.fim_rep_id != LLM_TOKEN_NULL) special_fim_rep_id = vocab_data.fim_rep_id;
        for (const auto & it : special_token_types) {
            int32_t & id = std::get<1>(it);            
            if (id != LLM_TOKEN_NULL && id >= id_to_token.size()) {
                LLM_LOG(true, "%s: bad special token\n", __func__);
            }
        }

        // Handle add_bos, add_eos and add_sep
        {
            if (vocab_data.read_add_bos) add_bos = vocab_data.add_bos;
            if (vocab_data.read_add_eos) add_eos = vocab_data.add_eos;
            if (vocab_data.read_add_sep) add_sep = vocab_data.add_sep;
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text

        for (const auto & t : token_to_id) {
            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if (special_eot_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|eot_id|>"
                        || t.first == "<|im_end|>"
                        || t.first == "<|end|>"
                        || t.first == "<end_of_turn>"
                        || t.first == "<|endoftext|>"
                        || t.first == "<EOT>"
                        || t.first == "_<EOT>"
                        || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                        || t.first == "<end_of_utterance>" // smoldocling
                   ) {
                    special_eot_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find EOM token: "<|eom_id|>"
            if (special_eom_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|eom_id|>"
                        ) {
                    special_eom_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if (special_fim_pre_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_prefix|>"  // Qwen
                        || t.first == "<fim-prefix>"
                        || t.first == "<fim_prefix>"    // Granite
                        || t.first == "<｜fim▁begin｜>" // DeepSeek
                        || t.first == "<PRE>"
                        || t.first == "▁<PRE>"          // CodeLlama
                        || t.first == "<|code_prefix|>" // GLM-4.5
                        ) {
                    special_fim_pre_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if (special_fim_suf_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_suffix|>" // Qwen
                        || t.first == "<fim-suffix>"
                        || t.first == "<fim_suffix>"   // Granite
                        || t.first == "<｜fim▁hole｜>" // DeepSeek
                        || t.first == "<SUF>"
                        || t.first == "▁<SUF>"         // CodeLlama
                        || t.first == "<|code_suffix|>" // GLM-4.5
                        ) {
                    special_fim_suf_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if (special_fim_mid_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_middle|>" // Qwen
                        || t.first == "<fim-middle>"
                        || t.first == "<fim_middle>"   // Granite
                        || t.first == "<｜fim▁end｜>"  // DeepSeek
                        || t.first == "<MID>"
                        || t.first == "▁<MID>"         // CodeLlama
                        || t.first == "<|code_middle|>" // GLM-4.5
                        ) {
                    special_fim_mid_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if (special_fim_pad_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_pad|>" // Qwen
                        || t.first == "<fim-pad>"
                        || t.first == "<fim_pad>"   // Granite
                        || t.first == "<PAD>"
                        ) {
                    special_fim_pad_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if (special_fim_rep_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_repo|>"  // Qwen
                        || t.first == "<|repo_name|>"
                        || t.first == "<fim-repo>"
                        || t.first == "<REPO>"
                        || t.first == "<reponame>"    // Granite
                        ) {
                    special_fim_rep_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if (special_fim_sep_id == LLM_TOKEN_NULL) {
                if (false
                        || t.first == "<|file_sep|>" // Qwen
                        ) {
                    special_fim_sep_id = t.second;
                    if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                        LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                    }
                }
            }
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        special_eog_ids.clear();

        if (special_fim_pad_id != LLM_TOKEN_NULL && special_eog_ids.count(special_fim_pad_id) == 0) {
            special_eog_ids.insert(special_fim_pad_id);
        }

        if (special_fim_rep_id != LLM_TOKEN_NULL && special_eog_ids.count(special_fim_rep_id) == 0) {
            special_eog_ids.insert(special_fim_rep_id);
        }

        if (special_fim_sep_id != LLM_TOKEN_NULL && special_eog_ids.count(special_fim_sep_id) == 0) {
            special_eog_ids.insert(special_fim_sep_id);
        }

        for (const auto & t : token_to_id) {
            if (false
                    || t.first == "<|eot_id|>"
                    || t.first == "<|im_end|>"
                    || t.first == "<|end|>"
                    || t.first == "<|return|>" // o200k_harmony
                    || t.first == "<|call|>"   // o200k_harmony
                    || t.first == "<end_of_turn>"
                    || t.first == "<|endoftext|>"
                    || t.first == "<|eom_id|>"
                    || t.first == "<EOT>"
                    || t.first == "_<EOT>"
                    || t.first == "<|end_of_text|>"
                    || t.first == "<end_of_utterance>" // smoldocling
               ) {
                special_eog_ids.insert(t.second);
                if ((id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL) == 0) {
                    LLM_LOG(true, "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                    id_to_token[t.second].attr = LLM_TOKEN_ATTR_CONTROL;
                }
            } else {
                // token is control, but not marked as EOG -> print a debug log
                if (id_to_token[t.second].attr & LLM_TOKEN_ATTR_CONTROL && special_eog_ids.count(t.second) == 0) {
                    LLM_LOG(false, "%s: control token: %6d '%s' is not marked as EOG\n",
                            __func__, t.second, t.first.c_str());
                }
            }
        }

        // @ngxson : quick hack for gpt-oss, always render these tokens
        for (const auto & t : token_to_id) {
            if (t.first == "<|channel|>" || t.first == "<|message|>" || t.first == "<|start|>" || t.first == "<|constrain|>") {
                id_to_token[t.second].attr = LLM_TOKEN_ATTR_USER_DEFINED;
            }
        }

        // sanity checks
        if (special_eos_id != LLM_TOKEN_NULL && special_eog_ids.count(special_eos_id) == 0) {
            special_eog_ids.insert(special_eos_id);
            LLM_LOG(true, "%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eot_id != LLM_TOKEN_NULL && special_eog_ids.count(special_eot_id) == 0) {
            special_eog_ids.insert(special_eot_id);
            LLM_LOG(true, "%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eom_id != LLM_TOKEN_NULL && special_eog_ids.count(special_eom_id) == 0) {
            special_eog_ids.insert(special_eom_id);
            LLM_LOG(true, "%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        // TODO: workaround for o200k_harmony tokenizer: the "<|end|>" token should not be EOG
        //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens,
        //       we remove the "<|end|>" token from the EOG list
        {
            bool has_return = false;
            bool has_call   = false;
            bool has_end    = false;

            llm_token end_id = LLM_TOKEN_NULL;

            LLM_LOG(false, "%s: printing all EOG tokens:\n", __func__);
            for (auto tid : special_eog_ids) {
                LLM_LOG(false, "%s:   - %d ('%s')\n", __func__, tid, id_to_token[tid].text.c_str());

                if (id_to_token[tid].text == "<|return|>") {
                    has_return = true;
                } else if (id_to_token[tid].text == "<|call|>") {
                    has_call = true;
                } else if (id_to_token[tid].text == "<|end|>") {
                    has_end = true;
                    end_id = tid;
                }
            }

            if (has_return && has_call && has_end) {
                special_eog_ids.erase(end_id);
                id_to_token[end_id].attr = LLM_TOKEN_ATTR_USER_DEFINED;
                LLM_LOG(false, "%s: special_eog_ids contains both '<|return|>' and '<|call|>' tokens, removing '<|end|>' token from EOG list\n", __func__);
            }
        }
    }

    // build special tokens cache
    {
        for (llm_token id = 0; id < (llm_token) n_tokens; ++id) {
            if (id_to_token[id].attr & (LLM_TOKEN_ATTR_CONTROL | LLM_TOKEN_ATTR_USER_DEFINED | LLM_TOKEN_ATTR_UNKNOWN)) {
                cache_special_tokens.push_back(id);
            }
        }

        std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
            [&] (const llm_token a, const llm_token b) {
                return id_to_token[a].text.size() > id_to_token[b].text.size();
            }
        );

        LLM_LOG(false, "%s: special tokens cache size = %u\n", __func__, (uint32_t) cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<std::string> cache(n_tokens);

        for (uint32_t id = 0; id < n_tokens; ++id) {
            cache[id] = token_to_piece_for_cache(id, true);

            size_cache += cache[id].size();
        }

        std::swap(cache_token_to_piece, cache);

        LLM_LOG(false, "%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the NNUF file.
    //TODO: Extract attributes from NNUF file.
    {
        auto _contains_any = [] (const std::string & str, const std::vector<std::string_view> & substrs) -> bool {
            for (const auto & substr : substrs) {
                if (str.find(substr) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };

        auto _set_tokenid_attr = [&] (const llm_token id, llm_token_attr attr, bool value) {
            uint32_t current = id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            id_to_token[id].attr = (llm_token_attr) current;
        };

        auto _set_token_attr = [&] (const std::string & token, llm_token_attr attr, bool value) {
            _set_tokenid_attr(token_to_id.at(token), attr, value);
        };

        std::string model_name = vocab_data.model;
        std::string tokenizer_pre = vocab_data.pre_model;
        std::string general_arch = "N/A";

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [] (const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer/architecture name
        if (false
                || _contains_any(tokenizer_pre, {"jina-v2-de", "jina-v2-es", "jina-v2-code"})
                || _contains_any(general_arch, {"nomic-bert-moe", "jina-bert-v3"})
           ) {
            if (token_to_id.count("<mask>") == 0) {
                LLM_LOG(true, "%s: Mask token is missing in vocab, please reconvert model!\n", __func__);
            } else {
                _set_token_attr("<mask>", LLM_TOKEN_ATTR_LSTRIP, true);
            }
        } else if (_contains_any(model_name, {"phi-3", "phi3"})) {
            for (auto id : cache_special_tokens) {
                _set_tokenid_attr(id, LLM_TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"</s>"}) {
                _set_token_attr(token, LLM_TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"<unk>", "<s>", "<|endoftext|>"}) {
                _set_token_attr(token, LLM_TOKEN_ATTR_RSTRIP, false);
            }
        }
    }
}

enum llm_tokenizer_type llm_vocab::impl::get_type() const {
    return type;
}

std::string llm_vocab::impl::type_name() const{
    switch (type) {
        case LLM_TOKENIZER_TYPE_NONE:   return "no vocab";
        case LLM_TOKENIZER_TYPE_SPM:    return "SPM";
        case LLM_TOKENIZER_TYPE_BPE:    return "BPE";
        case LLM_TOKENIZER_TYPE_WPM:    return "WPM";
        case LLM_TOKENIZER_TYPE_UGM:    return "UGM";
        case LLM_TOKENIZER_TYPE_RWKV:   return "RWKV";
        case LLM_TOKENIZER_TYPE_PLAMO2: return "PLaMo2";
        default:                      return "unknown";
    }
}

bool llm_vocab::impl::is_normal(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token[id].attr & LLM_TOKEN_ATTR_NORMAL;
}

bool llm_vocab::impl::is_unknown(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token[id].attr & LLM_TOKEN_ATTR_UNKNOWN;
}

bool llm_vocab::impl::is_control(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token[id].attr & LLM_TOKEN_ATTR_CONTROL;
}

bool llm_vocab::impl::is_byte(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token[id].attr & LLM_TOKEN_ATTR_BYTE;
}

bool llm_vocab::impl::is_user_defined(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token[id].attr & LLM_TOKEN_ATTR_USER_DEFINED;
}

bool llm_vocab::impl::is_unused(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token[id].attr & LLM_TOKEN_ATTR_UNUSED;
}

bool llm_vocab::impl::is_eog(llm_token id) const {
    return id != LLM_TOKEN_NULL && special_eog_ids.count(id) > 0;
}

uint8_t llm_vocab::impl::token_to_byte(llm_token id) const {
    assert(get_type() != LLM_TOKENIZER_TYPE_NONE);
    assert(is_byte(id));
    const auto & token_data = id_to_token.at(id);
    switch (get_type()) {
        case LLM_TOKENIZER_TYPE_SPM:
        case LLM_TOKENIZER_TYPE_UGM: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case LLM_TOKENIZER_TYPE_BPE: {
            throw std::runtime_error("fatal error");
        }
        case LLM_TOKENIZER_TYPE_WPM: {
            throw std::runtime_error("fatal error");
        }
        default:
            throw std::runtime_error("fatal error");
    }
}

llm_token_attr llm_vocab::impl::token_get_attr(llm_token id) const {
    assert(type != LLM_TOKENIZER_TYPE_NONE);
    return id_to_token.at(id).attr;
}

void llm_vocab::impl::init_tokenizer(enum llm_tokenizer_type type) {
    LLM_LOG(false, "%s: initializing tokenizer for type %d\n", __func__, type);

    switch (type) {
        case LLM_TOKENIZER_TYPE_SPM:
            tokenizer = std::make_unique<llm_tokenizer_spm>(vocab);
            break;
        case LLM_TOKENIZER_TYPE_BPE:
            tokenizer = std::make_unique<llm_tokenizer_bpe>(vocab);
            break;
        case LLM_TOKENIZER_TYPE_WPM:
            tokenizer = std::make_unique<llm_tokenizer_wpm>(vocab);
            break;
        case LLM_TOKENIZER_TYPE_UGM:
            tokenizer = std::make_unique<llm_tokenizer_ugm>(vocab, precompiled_charsmap);
            break;
        case LLM_TOKENIZER_TYPE_RWKV:
            tokenizer = std::make_unique<llm_tokenizer_rwkv>(vocab);
            break;
        case LLM_TOKENIZER_TYPE_PLAMO2:
            tokenizer = std::make_unique<llm_tokenizer_plamo2>(vocab);
            break;
        default:
            throw std::runtime_error("unsupported vocab type");
    }
}


// (de-) tokenize
// #define PRETOKENIZERDEBUG

void llm_vocab::impl::tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const {
    // for each special token
    for (const llm_token special_id : cache_special_tokens) {
        const auto & data = vocab.get_token_data(special_id);
        const auto & text = data.text;

        if (!parse_special && (data.attr & (LLM_TOKEN_ATTR_CONTROL | LLM_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    //  string_view begins at pos 0 for the same reason
                    auto match = std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(text, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

#ifdef PRETOKENIZERDEBUG
                    LLM_LOG(true, "FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLM_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + text.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        if (data.attr & LLM_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

// NOTE: avoid ever using this except for building the token_to_piece caches
std::string llm_vocab::impl::token_to_piece_for_cache(llm_token token, bool special) const {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
        assert(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    assert(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    assert(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

static void llm_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llm_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

static std::string llm_decode_text(const std::string & text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

std::vector<llm_token> llm_vocab::impl::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    assert(tokenizer && "Tokenizer not initialized. Call llm_vocab::init_tokenizer() first.");

    std::vector<llm_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer, parse_special);
    }

    switch (get_type()) {
        case LLM_TOKENIZER_TYPE_SPM:
            {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []

                bool is_prev_special = true;  // prefix with space if first token

                if (add_special && add_bos) {
                    assert(special_bos_id != LLM_TOKEN_NULL);
                    output.push_back(special_bos_id);
                    is_prev_special = true;
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text;

                        // prefix with space if previous is special
                        if (add_space_prefix && is_prev_special) {
                            text = ' ';
                        }

                        text += fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        llm_escape_whitespace(text);
                        llm_tokenizer_spm_session session(vocab);
                        session.tokenize(text, output);
                        is_prev_special = false;
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                        is_prev_special = true;
                    }
                }

                if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
                    LLM_LOG(true, 
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && add_eos) {
                    assert(special_eos_id != LLM_TOKEN_NULL);
                    output.push_back(special_eos_id);
                }
            } break;
        case LLM_TOKENIZER_TYPE_BPE:
            {
                llm_tokenizer_bpe_session session(vocab, *static_cast<const llm_tokenizer_bpe *>(tokenizer.get()));
                // it calls some other methods that are not exist in llm_tokenizer,
                // here just cast it to bpe tokenizer object
                if (add_special) {
                    session.append_bos(output);
                }
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        session.append(fragment.token, output);
                    }
                }

                if (add_special) {
                    session.append_eos(output);
                    session.check_double_bos_eos(output);
                }
            } break;
        case LLM_TOKENIZER_TYPE_WPM:
            {
                if (add_special) {
                    assert(special_bos_id != LLM_TOKEN_NULL);
                    output.push_back(special_bos_id);
                }

                llm_tokenizer_wpm_session session(vocab);

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special) {
                    assert(special_sep_id != LLM_TOKEN_NULL);
                    output.push_back(special_sep_id);
                }
            } break;
        case LLM_TOKENIZER_TYPE_UGM:
            {
                if (add_special && add_bos) {
                    assert(special_bos_id != LLM_TOKEN_NULL);
                    output.push_back(special_bos_id);
                }
                llm_tokenizer_ugm_session session(vocab, *static_cast<const llm_tokenizer_ugm *>(tokenizer.get()));

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);
#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
                    LLM_LOG(true, 
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && add_eos) {
                    assert(special_eos_id != LLM_TOKEN_NULL);
                    output.push_back(special_eos_id);
                }
            } break;
        case LLM_TOKENIZER_TYPE_RWKV:
            {
                llm_tokenizer_rwkv_session session(vocab, *static_cast<const llm_tokenizer_rwkv *>(tokenizer.get()));
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLM_TOKENIZER_TYPE_PLAMO2:
            {
                llm_tokenizer_plamo2_session session(*static_cast<const llm_tokenizer_plamo2 *>(tokenizer.get()));
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLM_LOG(true, "TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLM_TOKENIZER_TYPE_NONE:
            throw std::runtime_error("fatal error");
    }

    return output;
}

int32_t llm_vocab::impl::token_to_piece(llm_token token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    static const int attr_special = LLM_TOKEN_ATTR_UNKNOWN | LLM_TOKEN_ATTR_CONTROL;
    const llm_token_attr attr = token_get_attr(token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=] (const char * token, size_t size) -> int32_t {
        if (size >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("invalid token size: exceeds int32_t limit");
        }

        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t) size;
        }
        memcpy(buf, token, size);
        return (int32_t) size;
    };

    // if we have a cache - use it
    {
        const auto & cache = cache_token_to_piece;

        if (!cache.empty()) {
            const auto & result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t) id_to_token.size()) {
        const std::string & token_text = id_to_token[token].text;
        switch (get_type()) {
            case LLM_TOKENIZER_TYPE_WPM:
            case LLM_TOKENIZER_TYPE_SPM:
            case LLM_TOKENIZER_TYPE_UGM: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLM_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & LLM_TOKEN_ATTR_NORMAL) {
                    std::string result = token_text;
                    llm_unescape_whitespace(result);
                    return _try_copy(result.data(), result.size());
                }
                if (attr & LLM_TOKEN_ATTR_BYTE) {
                    char byte = (char) token_to_byte(token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case LLM_TOKENIZER_TYPE_BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLM_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & LLM_TOKEN_ATTR_NORMAL) {
                    std::string result = llm_decode_text(token_text);
                    return _try_copy(result.data(), result.size());
                }
                break;
            }
            case LLM_TOKENIZER_TYPE_RWKV: {
                std::vector<uint8_t> result = llm_unescape_rwkv_token(token_text);

                // If we don't have enough space, return an error
                if (result.size() > (size_t)length) {
                    return -(int)result.size();
                }

                memcpy(buf, result.data(), result.size());
                return (int)result.size();
            }
            case LLM_TOKENIZER_TYPE_PLAMO2: {
                // PLaMo-2 uses similar token handling as BPE/SPM
                if (vocab.is_byte(token)) {
                    // Handle byte tokens like <0xXX>
                    if (token_text.length() == 6 && token_text.substr(0, 3) == "<0x" && token_text.back() == '>') {
                        int hex_val = std::stoi(token_text.substr(3, 2), nullptr, 16);
                        if (length < 1) {
                            return -1;
                        }
                        buf[0] = static_cast<char>(hex_val);
                        return 1;
                    }
                }

                // Normal token - just copy the text
                std::string result = token_text;
                return _try_copy(result.data(), result.size());
            }
            default:
                throw std::runtime_error("fatal error");
        }
    }

    return 0;
}

const std::string & llm_vocab::impl::token_to_piece(llm_token token) const {
    return cache_token_to_piece.at(token);
}

int32_t llm_vocab::impl::detokenize(
               const llm_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    if (type == LLM_TOKENIZER_TYPE_NONE) {
        return 0;
    }

    assert(tokenizer && "Tokenizer not initialized. Call llm_vocab::init_tokenizer() first.");

    int32_t avail = text_len_max;
    int32_t total = 0;

    // remove the leading space
    bool remove_space = add_space_prefix;

    if (remove_special && add_bos) {
        if (n_tokens > 0 && tokens[0] == special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && add_eos) {
        if (n_tokens > 0 && tokens[n_tokens - 1] == special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        assert(avail >= 0);
        int32_t n_chars = token_to_piece(tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) {
            avail = 0;
            total -= n_chars;
        } else if (n_chars > 0) {
            avail -= n_chars;
            text  += n_chars;
            total += n_chars;
        }
    }

    if (total > text_len_max) {
        return -total;
    }

    if (clean_spaces) {
        text -= total;  // restart text

        // first pass: characters ?!.,  //TODO: where do these characters come from?
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '?' || x == '!' || x == '.' || x == ',') {  // " ?", " !", " .", " ,"
                    total--;  // remove space
                }
            }
            text[total++] = x;
        }

        // second pass: strip single apostrophe between spaces
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {  // " ' "
                total--;           // remove prev space
                text[++i] = '\0';  // remove next space
            }
            text[total++] = x;
        }

        // third pass: apostrophe contractions  //NOTE: this makes sense?
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '\'' && i + 1 < total3) {
                    const char x1 = text[i + 1];
                    if (x1 == 't' || x1 == 'd') {  // " 't", " 'd"
                        //total--;  // remove space
                    } else if (x1 == 's' || x1 == 'm') {  // " 's", " 'm"
                        total--;  // remove space
                    } else if (i + 2 < total3) {
                        const char x2 = text[i + 2];
                        if ((x1 == 'l' && x2 == 'l')) {  // " 'll"
                            //total--;  // remove space
                        } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {  // " 're", " 've"
                            total--;  // remove space
                        } else {
                            //total--;  // remove space
                        }
                    } else {
                        //total--;  // remove space
                    }
                }
            }
            text[total++] = x;
        }
    }

    return total <= text_len_max ? total : -total;
}

void llm_vocab::impl::print_info() const {
    LLM_LOG(true, "%s: vocab type       = %s\n",     __func__, type_name().c_str());
    LLM_LOG(true, "%s: n_vocab          = %u\n",     __func__, vocab.n_tokens());
    LLM_LOG(true, "%s: n_merges         = %u\n",     __func__, (uint32_t) bpe_ranks.size());

    // special tokens
    if (special_bos_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: BOS token        = %d '%s'\n", __func__, special_bos_id,     id_to_token.at(special_bos_id).text.c_str() );  }
    if (special_eos_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: EOS token        = %d '%s'\n", __func__, special_eos_id,     id_to_token.at(special_eos_id).text.c_str() );  }
    if (special_eot_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: EOT token        = %d '%s'\n", __func__, special_eot_id,     id_to_token.at(special_eot_id).text.c_str() );  }
    if (special_eom_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: EOM token        = %d '%s'\n", __func__, special_eom_id,     id_to_token.at(special_eom_id).text.c_str() );  }
    if (special_unk_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: UNK token        = %d '%s'\n", __func__, special_unk_id,     id_to_token.at(special_unk_id).text.c_str() );  }
    if (special_sep_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: SEP token        = %d '%s'\n", __func__, special_sep_id,     id_to_token.at(special_sep_id).text.c_str() );  }
    if (special_pad_id  != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: PAD token        = %d '%s'\n", __func__, special_pad_id,     id_to_token.at(special_pad_id).text.c_str() );  }
    if (special_mask_id != LLM_TOKEN_NULL)    { LLM_LOG(true,  "%s: MASK token       = %d '%s'\n", __func__, special_mask_id,    id_to_token.at(special_mask_id).text.c_str() ); }

    if (linefeed_id != LLM_TOKEN_NULL)        { LLM_LOG(true,  "%s: LF token         = %d '%s'\n", __func__, linefeed_id,        id_to_token.at(linefeed_id).text.c_str() ); }

    if (special_fim_pre_id != LLM_TOKEN_NULL) { LLM_LOG(true,  "%s: FIM PRE token    = %d '%s'\n", __func__, special_fim_pre_id, id_to_token.at(special_fim_pre_id).text.c_str() ); }
    if (special_fim_suf_id != LLM_TOKEN_NULL) { LLM_LOG(true,  "%s: FIM SUF token    = %d '%s'\n", __func__, special_fim_suf_id, id_to_token.at(special_fim_suf_id).text.c_str() ); }
    if (special_fim_mid_id != LLM_TOKEN_NULL) { LLM_LOG(true,  "%s: FIM MID token    = %d '%s'\n", __func__, special_fim_mid_id, id_to_token.at(special_fim_mid_id).text.c_str() ); }
    if (special_fim_pad_id != LLM_TOKEN_NULL) { LLM_LOG(true,  "%s: FIM PAD token    = %d '%s'\n", __func__, special_fim_pad_id, id_to_token.at(special_fim_pad_id).text.c_str() ); }
    if (special_fim_rep_id != LLM_TOKEN_NULL) { LLM_LOG(true,  "%s: FIM REP token    = %d '%s'\n", __func__, special_fim_rep_id, id_to_token.at(special_fim_rep_id).text.c_str() ); }
    if (special_fim_sep_id != LLM_TOKEN_NULL) { LLM_LOG(true,  "%s: FIM SEP token    = %d '%s'\n", __func__, special_fim_sep_id, id_to_token.at(special_fim_sep_id).text.c_str() ); }

    for (const auto & id : special_eog_ids) {
        LLM_LOG(true,  "%s: EOG token        = %d '%s'\n", __func__, id, id_to_token.at(id).text.c_str() );
    }

    LLM_LOG(true, "%s: max token length = %d\n", __func__, max_token_len);
}

llm_vocab::llm_vocab() : pimpl(new impl(*this)) {
}

llm_vocab::~llm_vocab() {
}

void llm_vocab::load(vocab_data & vocab_data) {
    pimpl->load(vocab_data);
}

std::string llm_vocab::get_tokenizer_model() const {
    return pimpl->tokenizer_model;
}

std::string llm_vocab::get_tokenizer_pre() const {
    return pimpl->tokenizer_pre;
}

enum llm_tokenizer_type llm_vocab::get_type() const {
    return pimpl->type;
}

enum llm_tokenizer_pre_type llm_vocab::get_pre_type() const {
    return pimpl->pre_type;
}

uint32_t llm_vocab::n_tokens() const {
    return (uint32_t) pimpl->id_to_token.size();
}

uint32_t llm_vocab::n_token_types() const {
    return (uint32_t) pimpl->n_token_types;
}

std::string llm_vocab::type_name() const{
    return pimpl->type_name();
}

bool llm_vocab::is_normal(llm_token id) const {
    return pimpl->is_normal(id);
}

bool llm_vocab::is_unknown(llm_token id) const {
    return pimpl->is_unknown(id);
}

bool llm_vocab::is_control(llm_token id) const {
    return pimpl->is_control(id);
}

bool llm_vocab::is_byte(llm_token id) const {
    return pimpl->is_byte(id);
}

bool llm_vocab::is_user_defined(llm_token id) const {
    return pimpl->is_user_defined(id);
}

bool llm_vocab::is_unused(llm_token id) const {
    return pimpl->is_unused(id);
}

bool llm_vocab::is_eog(llm_token id) const {
    return pimpl->is_eog(id);
}

uint8_t llm_vocab::token_to_byte(llm_token id) const {
    return pimpl->token_to_byte(id);
}

llm_token llm_vocab::byte_to_token(uint8_t ch) const {
    assert(get_type() != LLM_TOKENIZER_TYPE_NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (get_type()) {
        case LLM_TOKENIZER_TYPE_SPM:
        case LLM_TOKENIZER_TYPE_UGM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = pimpl->token_to_id.find(buf);
            if (token != pimpl->token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return pimpl->token_to_id.at(buf2);
        }
        case LLM_TOKENIZER_TYPE_WPM:
        case LLM_TOKENIZER_TYPE_BPE: {
            return pimpl->token_to_id.at(unicode_byte_to_utf8(ch));
        }
        case LLM_TOKENIZER_TYPE_PLAMO2: {
            // PLaMo-2 uses byte tokens in format <0xXX>
            char hex_str[8];
            snprintf(hex_str, sizeof(hex_str), "<0x%02X>", ch);
            return pimpl->token_to_id.at(hex_str);
        }
        default:
            throw std::runtime_error("fatal error");
    }
}

llm_token llm_vocab::text_to_token(const std::string & text) const {
    assert(pimpl->type != LLM_TOKENIZER_TYPE_NONE);
    auto it = pimpl->token_to_id.find(text);
    if (it != pimpl->token_to_id.end()) {
        return (*it).second;
    }
    return LLM_TOKEN_NULL;
}

const llm_vocab::token_data & llm_vocab::get_token_data(llm_token id) const {
    assert(pimpl->type != LLM_TOKENIZER_TYPE_NONE);
    return pimpl->id_to_token.at(id);
}

const char * llm_vocab::token_get_text(llm_token id) const {
    assert(pimpl->type != LLM_TOKENIZER_TYPE_NONE);
    return pimpl->id_to_token.at(id).text.c_str();
}

float llm_vocab::token_get_score(llm_token id) const {
    assert(pimpl->type != LLM_TOKENIZER_TYPE_NONE);
    return pimpl->id_to_token.at(id).score;
}

llm_token_attr llm_vocab::token_get_attr(llm_token id) const {
    return pimpl->token_get_attr(id);
}

llm_token llm_vocab::token_bos() const {
    return pimpl->special_bos_id;
}

llm_token llm_vocab::token_eos() const {
    return pimpl->special_eos_id;
}

llm_token llm_vocab::token_eot() const {
    return pimpl->special_eot_id;
}

llm_token llm_vocab::token_eom() const {
    return pimpl->special_eom_id;
}

llm_token llm_vocab::token_unk() const {
    return pimpl->special_unk_id;
}

llm_token llm_vocab::token_sep() const {
    return pimpl->special_sep_id;
}

llm_token llm_vocab::token_nl() const {
    return pimpl->linefeed_id;
}

llm_token llm_vocab::token_pad() const {
    return pimpl->special_pad_id;
}

llm_token llm_vocab::token_prefix() const {
    return pimpl->special_fim_pre_id;
}

llm_token llm_vocab::token_middle() const {
    return pimpl->special_fim_mid_id;
}

llm_token llm_vocab::token_suffix() const {
    return pimpl->special_fim_suf_id;
}

llm_token llm_vocab::token_fim_pre() const {
    return pimpl->special_fim_pre_id;
}

llm_token llm_vocab::token_fim_suf() const {
    return pimpl->special_fim_suf_id;
}

llm_token llm_vocab::token_fim_mid() const {
    return pimpl->special_fim_mid_id;
}

llm_token llm_vocab::token_fim_pad() const {
    return pimpl->special_fim_pad_id;
}

llm_token llm_vocab::token_fim_rep() const {
    return pimpl->special_fim_rep_id;
}

llm_token llm_vocab::token_fim_sep() const {
    return pimpl->special_fim_sep_id;
}

llm_token llm_vocab::token_mask() const {
    return pimpl->special_mask_id;
}

bool llm_vocab::get_add_space_prefix() const {
    return pimpl->add_space_prefix;
}

bool llm_vocab::get_add_bos() const {
    return pimpl->add_bos;
}

bool llm_vocab::get_add_eos() const {
    return pimpl->add_eos;
}

bool llm_vocab::get_add_sep() const {
    return pimpl->add_sep;
}

bool llm_vocab::get_ignore_merges() const {
    return pimpl->ignore_merges;
}

bool llm_vocab::get_clean_spaces() const {
    return pimpl->clean_spaces;
}

bool llm_vocab::get_remove_extra_whitespaces() const {
    return pimpl->remove_extra_whitespaces;
}

bool llm_vocab::get_escape_whitespaces() const {
    return pimpl->escape_whitespaces;
}

bool llm_vocab::get_treat_whitespace_as_suffix() const {
    return pimpl->treat_whitespace_as_suffix;
}

int llm_vocab::max_token_len() const {
    return pimpl->max_token_len;
}

int llm_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    assert(token_left.find(' ')   == std::string::npos);
    assert(token_left.find('\n')  == std::string::npos);
    assert(token_right.find(' ')  == std::string::npos);
    assert(token_right.find('\n') == std::string::npos);

    auto it = pimpl->bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == pimpl->bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

std::vector<std::string> llm_vocab::get_bpe_merges() const {
    std::vector<std::string> result(pimpl->bpe_ranks.size());

    for (const auto & pair : pimpl->bpe_ranks) {
        result[pair.second] = pair.first.first + " " + pair.first.second;
    }

    return result;
}

std::vector<char> llm_vocab::get_precompiled_charsmap() const {
    return pimpl->precompiled_charsmap;
}

int32_t llm_vocab::tokenize(
                  const char * text,
                     int32_t   text_len,
                 llm_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) const {
    auto res = tokenize(std::string(text, text_len), add_special, parse_special);
    if (res.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        LLM_ERROR("%s: tokenization result size %zu exceeds int32_t limit\n", __func__, res.size());
        return std::numeric_limits<int32_t>::min();
    }

    if (n_tokens_max < (int) res.size()) {
        // LLM_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

std::vector<llm_token> llm_vocab::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    return pimpl->tokenize(raw_text, add_special, parse_special);
}

const std::string & llm_vocab::token_to_piece(llm_token token) const {
    return pimpl->token_to_piece(token);
}

int32_t llm_vocab::token_to_piece(llm_token token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    return pimpl->token_to_piece(token, buf, length, lstrip, special);
}

int32_t llm_vocab::detokenize(
               const llm_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    return pimpl->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}

std::string llm_vocab::detokenize(const std::vector<llm_token> & tokens, bool special) const {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        assert(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

void llm_vocab::print_info() const {
    pimpl->print_info();
}


// interface implementation
int32_t llm_vocab_n_tokens(const struct llm_vocab * vocab) {
    return vocab->n_tokens();
}

// deprecated
int32_t llm_n_vocab(const struct llm_vocab * vocab) {
    return llm_vocab_n_tokens(vocab);
}

enum llm_tokenizer_type llm_vocab_type(const struct llm_vocab * vocab) {
    return vocab->get_type();
}

const char * llm_vocab_get_text(const struct llm_vocab * vocab, llm_token token) {
    return vocab->token_get_text(token);
}

float llm_vocab_get_score(const struct llm_vocab * vocab, llm_token token) {
    return vocab->token_get_score(token);
}

enum llm_token_attr llm_vocab_get_attr(const struct llm_vocab * vocab, llm_token token) {
    return vocab->token_get_attr(token);
}

bool llm_vocab_is_eog(const struct llm_vocab * vocab, llm_token token) {
    return vocab->is_eog(token);
}

bool llm_vocab_is_control(const struct llm_vocab * vocab, llm_token token) {
    return vocab->is_control(token);
}

llm_token llm_vocab_bos(const struct llm_vocab * vocab) {
    return vocab->token_bos();
}

llm_token llm_vocab_eos(const struct llm_vocab * vocab) {
    return vocab->token_eos();
}

llm_token llm_vocab_eot(const struct llm_vocab * vocab) {
    return vocab->token_eot();
}

// deprecated
llm_token llm_vocab_cls(const struct llm_vocab * vocab) {
    return vocab->token_bos();
}

llm_token llm_vocab_sep(const struct llm_vocab * vocab) {
    return vocab->token_sep();
}

llm_token llm_vocab_nl (const struct llm_vocab * vocab) {
    return vocab->token_nl();
}

llm_token llm_vocab_pad(const struct llm_vocab * vocab) {
    return vocab->token_pad();
}

bool llm_vocab_get_add_bos(const struct llm_vocab * vocab) {
    return vocab->get_add_bos();
}

bool llm_vocab_get_add_eos(const struct llm_vocab * vocab) {
    return vocab->get_add_eos();
}

bool llm_vocab_get_add_sep(const struct llm_vocab * vocab) {
    return vocab->get_add_sep();
}

llm_token llm_vocab_fim_pre(const struct llm_vocab * vocab) {
    return vocab->token_fim_pre();
}

llm_token llm_vocab_fim_suf(const struct llm_vocab * vocab) {
    return vocab->token_fim_suf();
}

llm_token llm_vocab_fim_mid(const struct llm_vocab * vocab) {
    return vocab->token_fim_mid();
}

llm_token llm_vocab_fim_pad(const struct llm_vocab * vocab) {
    return vocab->token_fim_pad();
}

llm_token llm_vocab_fim_rep(const struct llm_vocab * vocab) {
    return vocab->token_fim_rep();
}

llm_token llm_vocab_fim_sep(const struct llm_vocab * vocab) {
    return vocab->token_fim_sep();
}

llm_token llm_vocab_mask(const struct llm_vocab* vocab) {
    return vocab->token_mask();
}

// deprecated
const char * llm_token_get_text(const struct llm_vocab * vocab, llm_token token) {
    return llm_vocab_get_text(vocab, token);
}

// deprecated
float llm_token_get_score(const struct llm_vocab * vocab, llm_token token) {
    return llm_vocab_get_score(vocab, token);
}

// deprecated
enum llm_token_attr llm_token_get_attr(const struct llm_vocab * vocab, llm_token token) {
    return llm_vocab_get_attr(vocab, token);
}

// deprecated
bool llm_token_is_eog(const struct llm_vocab * vocab, llm_token token) {
    return llm_vocab_is_eog(vocab, token);
}

// deprecated
bool llm_token_is_control(const struct llm_vocab * vocab, llm_token token) {
    return llm_vocab_is_control(vocab, token);
}

// deprecated
llm_token llm_token_bos(const struct llm_vocab * vocab) {
    return llm_vocab_bos(vocab);
}

// deprecated
llm_token llm_token_eos(const struct llm_vocab * vocab) {
    return llm_vocab_eos(vocab);
}

// deprecated
llm_token llm_token_eot(const struct llm_vocab * vocab) {
    return llm_vocab_eot(vocab);
}

// deprecated
llm_token llm_token_cls(const struct llm_vocab * vocab) {
    //return llm_vocab_cls(vocab);
    return llm_vocab_bos(vocab); // avoid deprecation warning
}

// deprecated
llm_token llm_token_sep(const struct llm_vocab * vocab) {
    return llm_vocab_sep(vocab);
}

// deprecated
llm_token llm_token_nl (const struct llm_vocab * vocab) {
    return llm_vocab_nl(vocab);
}

// deprecated
llm_token llm_token_pad(const struct llm_vocab * vocab) {
    return llm_vocab_pad(vocab);
}

// deprecated
bool llm_add_bos_token(const struct llm_vocab * vocab) {
    return llm_vocab_get_add_bos(vocab);
}

// deprecated
bool llm_add_eos_token(const struct llm_vocab * vocab) {
    return llm_vocab_get_add_eos(vocab);
}

// deprecated
llm_token llm_token_fim_pre(const struct llm_vocab * vocab) {
    return llm_vocab_fim_pre(vocab);
}

// deprecated
llm_token llm_token_fim_suf(const struct llm_vocab * vocab) {
    return llm_vocab_fim_suf(vocab);
}

// deprecated
llm_token llm_token_fim_mid(const struct llm_vocab * vocab) {
    return llm_vocab_fim_mid(vocab);
}

// deprecated
llm_token llm_token_fim_pad(const struct llm_vocab * vocab) {
    return llm_vocab_fim_pad(vocab);
}

// deprecated
llm_token llm_token_fim_rep(const struct llm_vocab * vocab) {
    return llm_vocab_fim_rep(vocab);
}

// deprecated
llm_token llm_token_fim_sep(const struct llm_vocab * vocab) {
    return llm_vocab_fim_sep(vocab);
}


// tokenization
int32_t llm_tokenize(
    const struct llm_vocab * vocab,
                  const char * text,
                     int32_t   text_len,
                 llm_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    return vocab->tokenize(text, text_len, tokens, n_tokens_max, add_special, parse_special);
}

int32_t llm_token_to_piece(
    const struct llm_vocab * vocab,
                 llm_token   token,
                        char * buf,
                     int32_t   length,
                     int32_t   lstrip,
                        bool   special) {
    return vocab->token_to_piece(token, buf, length, lstrip, special);
}

int32_t llm_detokenize(
    const struct llm_vocab * vocab,
           const llm_token * tokens,
                     int32_t   n_tokens,
                        char * text,
                     int32_t   text_len_max,
                        bool   remove_special,
                        bool   unparse_special) {
    return vocab->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}


// Vocab utils
std::vector<llm_token> common_tokenize(
    const struct llm_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llm_token> result(n_tokens);
    n_tokens = llm_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llm_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        assert(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llm_vocab * vocab, llm_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llm_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llm_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        assert(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llm_vocab * vocab, const std::vector<llm_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llm_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llm_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        assert(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//==================== unicode utils =====================//

uint32_t unicode_tolower(uint32_t cpt) {
    // binary search
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}


std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        }
        catch (const std::invalid_argument & /*ex*/) {
            // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llm_tokenize
            ++offset;
            result.emplace_back(0xFFFD); // replacement character
        }
    }
    return result;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            // regex: <space>?\p{L}+
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        auto _get_flags = [&] (const size_t pos) -> unicode_cpt_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {

        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    }

    return bpe_offsets;
}

// use std::wregex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::wstring & wtext, const std::wstring & regex_expr, const std::vector<size_t> & offsets) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::wcmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// use std::regex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::regex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::cmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string & s) {
#if defined(__clang__)
    // disable C++17 deprecation warning for std::codecvt_utf8
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

    return conv.from_bytes(s);
}

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    // unicode categories
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", unicode_cpt_flags::NUMBER },
        { "\\p{L}", unicode_cpt_flags::LETTER },
        { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
        { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
        { "\\p{S}", unicode_cpt_flags::SYMBOL },
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },
        { unicode_cpt_flags::LETTER,      0xD2 },
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
        { unicode_cpt_flags::SYMBOL,      0xD5 },
    };

    static const std::map<int, std::string> k_ucat_map = {
        { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
        { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
        { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
        { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
        { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" }, // $+<=>^`|
    };

    // compute collapsed codepoints only if needed by at least one regex
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        // search for unicode categories
        for (const auto & ucat : k_ucat_enum) {
            if (std::string::npos != regex_expr.find(ucat.first)) {
                need_collapse = true;
                break;
            }
        }
    }

    const auto cpts = unicode_cpts_from_utf8(text);

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    std::string text_collapsed;
    if (need_collapse) {
        // collapse all unicode categories
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags_from_cpt(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                text_collapsed[i] = (char) 0x0B;    // <vertical tab> as whitespace fallback
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0; // fallback
            }
        }
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }

            if (use_collapsed) {
                // sanity-check that the original regex does not contain any non-ASCII characters
                const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex
                std::string regex_expr_collapsed;

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    if (regex_expr[i + 0] == '\\' && i + 4 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{' &&
                        regex_expr[i + 4] == '}') {
                        const std::string pat = regex_expr.substr(i, 5);
                        if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                            if (!inside) {
                                regex_expr_collapsed += '[';
                            }
                            regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                            regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                            if (!inside) {
                                regex_expr_collapsed += ']';
                            }
                            i += 4;
                            continue;
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                const std::wstring wregex_expr = unicode_wstring_from_utf8(regex_expr);

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1
{0x000000, 0x0080},
{0x000020, 0x0008},
{0x000021, 0x0020},
{0x000024, 0x0040},
{0x000025, 0x0020},
{0x00002B, 0x0040},
{0x00002C, 0x0020},
{0x000030, 0x0002},
{0x00003A, 0x0020},
{0x00003C, 0x0040},
{0x00003F, 0x0020},
{0x000041, 0x0004},
{0x00005B, 0x0020},
{0x00005E, 0x0040},
{0x00005F, 0x0020},
{0x000060, 0x0040},
{0x000061, 0x0004},
{0x00007B, 0x0020},
{0x00007C, 0x0040},
{0x00007D, 0x0020},
{0x00007E, 0x0040},
{0x00007F, 0x0080},
{0x0000A0, 0x0008},
{0x0000A1, 0x0020},
{0x0000A2, 0x0040},
{0x0000A7, 0x0020},
{0x0000A8, 0x0040},
{0x0000AA, 0x0004},
{0x0000AB, 0x0020},
{0x0000AC, 0x0040},
{0x0000AD, 0x0080},
{0x0000AE, 0x0040},
{0x0000B2, 0x0002},
{0x0000B4, 0x0040},
{0x0000B5, 0x0004},
{0x0000B6, 0x0020},
{0x0000B8, 0x0040},
{0x0000B9, 0x0002},
{0x0000BA, 0x0004},
{0x0000BB, 0x0020},
{0x0000BC, 0x0002},
{0x0000BF, 0x0020},
{0x0000C0, 0x0004},
{0x0000D7, 0x0040},
{0x0000D8, 0x0004},
{0x0000F7, 0x0040},
{0x0000F8, 0x0004},
{0x0002C2, 0x0040},
{0x0002C6, 0x0004},
{0x0002D2, 0x0040},
{0x0002E0, 0x0004},
{0x0002E5, 0x0040},
{0x0002EC, 0x0004},
{0x0002ED, 0x0040},
{0x0002EE, 0x0004},
{0x0002EF, 0x0040},
{0x000300, 0x0010},
{0x000370, 0x0004},
{0x000375, 0x0040},
{0x000376, 0x0004},
{0x000378, 0x0001},
{0x00037A, 0x0004},
{0x00037E, 0x0020},
{0x00037F, 0x0004},
{0x000380, 0x0001},
{0x000384, 0x0040},
{0x000386, 0x0004},
{0x000387, 0x0020},
{0x000388, 0x0004},
{0x00038B, 0x0001},
{0x00038C, 0x0004},
{0x00038D, 0x0001},
{0x00038E, 0x0004},
{0x0003A2, 0x0001},
{0x0003A3, 0x0004},
{0x0003F6, 0x0040},
{0x0003F7, 0x0004},
{0x000482, 0x0040},
{0x000483, 0x0010},
{0x00048A, 0x0004},
{0x000530, 0x0001},
{0x000531, 0x0004},
{0x000557, 0x0001},
{0x000559, 0x0004},
{0x00055A, 0x0020},
{0x000560, 0x0004},
{0x000589, 0x0020},
{0x00058B, 0x0001},
{0x00058D, 0x0040},
{0x000590, 0x0001},
{0x000591, 0x0010},
{0x0005BE, 0x0020},
{0x0005BF, 0x0010},
{0x0005C0, 0x0020},
{0x0005C1, 0x0010},
{0x0005C3, 0x0020},
{0x0005C4, 0x0010},
{0x0005C6, 0x0020},
{0x0005C7, 0x0010},
{0x0005C8, 0x0001},
{0x0005D0, 0x0004},
{0x0005EB, 0x0001},
{0x0005EF, 0x0004},
{0x0005F3, 0x0020},
{0x0005F5, 0x0001},
{0x000600, 0x0080},
{0x000606, 0x0040},
{0x000609, 0x0020},
{0x00060B, 0x0040},
{0x00060C, 0x0020},
{0x00060E, 0x0040},
{0x000610, 0x0010},
{0x00061B, 0x0020},
{0x00061C, 0x0080},
{0x00061D, 0x0020},
{0x000620, 0x0004},
{0x00064B, 0x0010},
{0x000660, 0x0002},
{0x00066A, 0x0020},
{0x00066E, 0x0004},
{0x000670, 0x0010},
{0x000671, 0x0004},
{0x0006D4, 0x0020},
{0x0006D5, 0x0004},
{0x0006D6, 0x0010},
{0x0006DD, 0x0080},
{0x0006DE, 0x0040},
{0x0006DF, 0x0010},
{0x0006E5, 0x0004},
{0x0006E7, 0x0010},
{0x0006E9, 0x0040},
{0x0006EA, 0x0010},
{0x0006EE, 0x0004},
{0x0006F0, 0x0002},
{0x0006FA, 0x0004},
{0x0006FD, 0x0040},
{0x0006FF, 0x0004},
{0x000700, 0x0020},
{0x00070E, 0x0001},
{0x00070F, 0x0080},
{0x000710, 0x0004},
{0x000711, 0x0010},
{0x000712, 0x0004},
{0x000730, 0x0010},
{0x00074B, 0x0001},
{0x00074D, 0x0004},
{0x0007A6, 0x0010},
{0x0007B1, 0x0004},
{0x0007B2, 0x0001},
{0x0007C0, 0x0002},
{0x0007CA, 0x0004},
{0x0007EB, 0x0010},
{0x0007F4, 0x0004},
{0x0007F6, 0x0040},
{0x0007F7, 0x0020},
{0x0007FA, 0x0004},
{0x0007FB, 0x0001},
{0x0007FD, 0x0010},
{0x0007FE, 0x0040},
{0x000800, 0x0004},
{0x000816, 0x0010},
{0x00081A, 0x0004},
{0x00081B, 0x0010},
{0x000824, 0x0004},
{0x000825, 0x0010},
{0x000828, 0x0004},
{0x000829, 0x0010},
{0x00082E, 0x0001},
{0x000830, 0x0020},
{0x00083F, 0x0001},
{0x000840, 0x0004},
{0x000859, 0x0010},
{0x00085C, 0x0001},
{0x00085E, 0x0020},
{0x00085F, 0x0001},
{0x000860, 0x0004},
{0x00086B, 0x0001},
{0x000870, 0x0004},
{0x000888, 0x0040},
{0x000889, 0x0004},
{0x00088F, 0x0001},
{0x000890, 0x0080},
{0x000892, 0x0001},
{0x000898, 0x0010},
{0x0008A0, 0x0004},
{0x0008CA, 0x0010},
{0x0008E2, 0x0080},
{0x0008E3, 0x0010},
{0x000904, 0x0004},
{0x00093A, 0x0010},
{0x00093D, 0x0004},
{0x00093E, 0x0010},
{0x000950, 0x0004},
{0x000951, 0x0010},
{0x000958, 0x0004},
{0x000962, 0x0010},
{0x000964, 0x0020},
{0x000966, 0x0002},
{0x000970, 0x0020},
{0x000971, 0x0004},
{0x000981, 0x0010},
{0x000984, 0x0001},
{0x000985, 0x0004},
{0x00098D, 0x0001},
{0x00098F, 0x0004},
{0x000991, 0x0001},
{0x000993, 0x0004},
{0x0009A9, 0x0001},
{0x0009AA, 0x0004},
{0x0009B1, 0x0001},
{0x0009B2, 0x0004},
{0x0009B3, 0x0001},
{0x0009B6, 0x0004},
{0x0009BA, 0x0001},
{0x0009BC, 0x0010},
{0x0009BD, 0x0004},
{0x0009BE, 0x0010},
{0x0009C5, 0x0001},
{0x0009C7, 0x0010},
{0x0009C9, 0x0001},
{0x0009CB, 0x0010},
{0x0009CE, 0x0004},
{0x0009CF, 0x0001},
{0x0009D7, 0x0010},
{0x0009D8, 0x0001},
{0x0009DC, 0x0004},
{0x0009DE, 0x0001},
{0x0009DF, 0x0004},
{0x0009E2, 0x0010},
{0x0009E4, 0x0001},
{0x0009E6, 0x0002},
{0x0009F0, 0x0004},
{0x0009F2, 0x0040},
{0x0009F4, 0x0002},
{0x0009FA, 0x0040},
{0x0009FC, 0x0004},
{0x0009FD, 0x0020},
{0x0009FE, 0x0010},
{0x0009FF, 0x0001},
{0x000A01, 0x0010},
{0x000A04, 0x0001},
{0x000A05, 0x0004},
{0x000A0B, 0x0001},
{0x000A0F, 0x0004},
{0x000A11, 0x0001},
{0x000A13, 0x0004},
{0x000A29, 0x0001},
{0x000A2A, 0x0004},
{0x000A31, 0x0001},
{0x000A32, 0x0004},
{0x000A34, 0x0001},
{0x000A35, 0x0004},
{0x000A37, 0x0001},
{0x000A38, 0x0004},
{0x000A3A, 0x0001},
{0x000A3C, 0x0010},
{0x000A3D, 0x0001},
{0x000A3E, 0x0010},
{0x000A43, 0x0001},
{0x000A47, 0x0010},
{0x000A49, 0x0001},
{0x000A4B, 0x0010},
{0x000A4E, 0x0001},
{0x000A51, 0x0010},
{0x000A52, 0x0001},
{0x000A59, 0x0004},
{0x000A5D, 0x0001},
{0x000A5E, 0x0004},
{0x000A5F, 0x0001},
{0x000A66, 0x0002},
{0x000A70, 0x0010},
{0x000A72, 0x0004},
{0x000A75, 0x0010},
{0x000A76, 0x0020},
{0x000A77, 0x0001},
{0x000A81, 0x0010},
{0x000A84, 0x0001},
{0x000A85, 0x0004},
{0x000A8E, 0x0001},
{0x000A8F, 0x0004},
{0x000A92, 0x0001},
{0x000A93, 0x0004},
{0x000AA9, 0x0001},
{0x000AAA, 0x0004},
{0x000AB1, 0x0001},
{0x000AB2, 0x0004},
{0x000AB4, 0x0001},
{0x000AB5, 0x0004},
{0x000ABA, 0x0001},
{0x000ABC, 0x0010},
{0x000ABD, 0x0004},
{0x000ABE, 0x0010},
{0x000AC6, 0x0001},
{0x000AC7, 0x0010},
{0x000ACA, 0x0001},
{0x000ACB, 0x0010},
{0x000ACE, 0x0001},
{0x000AD0, 0x0004},
{0x000AD1, 0x0001},
{0x000AE0, 0x0004},
{0x000AE2, 0x0010},
{0x000AE4, 0x0001},
{0x000AE6, 0x0002},
{0x000AF0, 0x0020},
{0x000AF1, 0x0040},
{0x000AF2, 0x0001},
{0x000AF9, 0x0004},
{0x000AFA, 0x0010},
{0x000B00, 0x0001},
{0x000B01, 0x0010},
{0x000B04, 0x0001},
{0x000B05, 0x0004},
{0x000B0D, 0x0001},
{0x000B0F, 0x0004},
{0x000B11, 0x0001},
{0x000B13, 0x0004},
{0x000B29, 0x0001},
{0x000B2A, 0x0004},
{0x000B31, 0x0001},
{0x000B32, 0x0004},
{0x000B34, 0x0001},
{0x000B35, 0x0004},
{0x000B3A, 0x0001},
{0x000B3C, 0x0010},
{0x000B3D, 0x0004},
{0x000B3E, 0x0010},
{0x000B45, 0x0001},
{0x000B47, 0x0010},
{0x000B49, 0x0001},
{0x000B4B, 0x0010},
{0x000B4E, 0x0001},
{0x000B55, 0x0010},
{0x000B58, 0x0001},
{0x000B5C, 0x0004},
{0x000B5E, 0x0001},
{0x000B5F, 0x0004},
{0x000B62, 0x0010},
{0x000B64, 0x0001},
{0x000B66, 0x0002},
{0x000B70, 0x0040},
{0x000B71, 0x0004},
{0x000B72, 0x0002},
{0x000B78, 0x0001},
{0x000B82, 0x0010},
{0x000B83, 0x0004},
{0x000B84, 0x0001},
{0x000B85, 0x0004},
{0x000B8B, 0x0001},
{0x000B8E, 0x0004},
{0x000B91, 0x0001},
{0x000B92, 0x0004},
{0x000B96, 0x0001},
{0x000B99, 0x0004},
{0x000B9B, 0x0001},
{0x000B9C, 0x0004},
{0x000B9D, 0x0001},
{0x000B9E, 0x0004},
{0x000BA0, 0x0001},
{0x000BA3, 0x0004},
{0x000BA5, 0x0001},
{0x000BA8, 0x0004},
{0x000BAB, 0x0001},
{0x000BAE, 0x0004},
{0x000BBA, 0x0001},
{0x000BBE, 0x0010},
{0x000BC3, 0x0001},
{0x000BC6, 0x0010},
{0x000BC9, 0x0001},
{0x000BCA, 0x0010},
{0x000BCE, 0x0001},
{0x000BD0, 0x0004},
{0x000BD1, 0x0001},
{0x000BD7, 0x0010},
{0x000BD8, 0x0001},
{0x000BE6, 0x0002},
{0x000BF3, 0x0040},
{0x000BFB, 0x0001},
{0x000C00, 0x0010},
{0x000C05, 0x0004},
{0x000C0D, 0x0001},
{0x000C0E, 0x0004},
{0x000C11, 0x0001},
{0x000C12, 0x0004},
{0x000C29, 0x0001},
{0x000C2A, 0x0004},
{0x000C3A, 0x0001},
{0x000C3C, 0x0010},
{0x000C3D, 0x0004},
{0x000C3E, 0x0010},
{0x000C45, 0x0001},
{0x000C46, 0x0010},
{0x000C49, 0x0001},
{0x000C4A, 0x0010},
{0x000C4E, 0x0001},
{0x000C55, 0x0010},
{0x000C57, 0x0001},
{0x000C58, 0x0004},
{0x000C5B, 0x0001},
{0x000C5D, 0x0004},
{0x000C5E, 0x0001},
{0x000C60, 0x0004},
{0x000C62, 0x0010},
{0x000C64, 0x0001},
{0x000C66, 0x0002},
{0x000C70, 0x0001},
{0x000C77, 0x0020},
{0x000C78, 0x0002},
{0x000C7F, 0x0040},
{0x000C80, 0x0004},
{0x000C81, 0x0010},
{0x000C84, 0x0020},
{0x000C85, 0x0004},
{0x000C8D, 0x0001},
{0x000C8E, 0x0004},
{0x000C91, 0x0001},
{0x000C92, 0x0004},
{0x000CA9, 0x0001},
{0x000CAA, 0x0004},
{0x000CB4, 0x0001},
{0x000CB5, 0x0004},
{0x000CBA, 0x0001},
{0x000CBC, 0x0010},
{0x000CBD, 0x0004},
{0x000CBE, 0x0010},
{0x000CC5, 0x0001},
{0x000CC6, 0x0010},
{0x000CC9, 0x0001},
{0x000CCA, 0x0010},
{0x000CCE, 0x0001},
{0x000CD5, 0x0010},
{0x000CD7, 0x0001},
{0x000CDD, 0x0004},
{0x000CDF, 0x0001},
{0x000CE0, 0x0004},
{0x000CE2, 0x0010},
{0x000CE4, 0x0001},
{0x000CE6, 0x0002},
{0x000CF0, 0x0001},
{0x000CF1, 0x0004},
{0x000CF3, 0x0010},
{0x000CF4, 0x0001},
{0x000D00, 0x0010},
{0x000D04, 0x0004},
{0x000D0D, 0x0001},
{0x000D0E, 0x0004},
{0x000D11, 0x0001},
{0x000D12, 0x0004},
{0x000D3B, 0x0010},
{0x000D3D, 0x0004},
{0x000D3E, 0x0010},
{0x000D45, 0x0001},
{0x000D46, 0x0010},
{0x000D49, 0x0001},
{0x000D4A, 0x0010},
{0x000D4E, 0x0004},
{0x000D4F, 0x0040},
{0x000D50, 0x0001},
{0x000D54, 0x0004},
{0x000D57, 0x0010},
{0x000D58, 0x0002},
{0x000D5F, 0x0004},
{0x000D62, 0x0010},
{0x000D64, 0x0001},
{0x000D66, 0x0002},
{0x000D79, 0x0040},
{0x000D7A, 0x0004},
{0x000D80, 0x0001},
{0x000D81, 0x0010},
{0x000D84, 0x0001},
{0x000D85, 0x0004},
{0x000D97, 0x0001},
{0x000D9A, 0x0004},
{0x000DB2, 0x0001},
{0x000DB3, 0x0004},
{0x000DBC, 0x0001},
{0x000DBD, 0x0004},
{0x000DBE, 0x0001},
{0x000DC0, 0x0004},
{0x000DC7, 0x0001},
{0x000DCA, 0x0010},
{0x000DCB, 0x0001},
{0x000DCF, 0x0010},
{0x000DD5, 0x0001},
{0x000DD6, 0x0010},
{0x000DD7, 0x0001},
{0x000DD8, 0x0010},
{0x000DE0, 0x0001},
{0x000DE6, 0x0002},
{0x000DF0, 0x0001},
{0x000DF2, 0x0010},
{0x000DF4, 0x0020},
{0x000DF5, 0x0001},
{0x000E01, 0x0004},
{0x000E31, 0x0010},
{0x000E32, 0x0004},
{0x000E34, 0x0010},
{0x000E3B, 0x0001},
{0x000E3F, 0x0040},
{0x000E40, 0x0004},
{0x000E47, 0x0010},
{0x000E4F, 0x0020},
{0x000E50, 0x0002},
{0x000E5A, 0x0020},
{0x000E5C, 0x0001},
{0x000E81, 0x0004},
{0x000E83, 0x0001},
{0x000E84, 0x0004},
{0x000E85, 0x0001},
{0x000E86, 0x0004},
{0x000E8B, 0x0001},
{0x000E8C, 0x0004},
{0x000EA4, 0x0001},
{0x000EA5, 0x0004},
{0x000EA6, 0x0001},
{0x000EA7, 0x0004},
{0x000EB1, 0x0010},
{0x000EB2, 0x0004},
{0x000EB4, 0x0010},
{0x000EBD, 0x0004},
{0x000EBE, 0x0001},
{0x000EC0, 0x0004},
{0x000EC5, 0x0001},
{0x000EC6, 0x0004},
{0x000EC7, 0x0001},
{0x000EC8, 0x0010},
{0x000ECF, 0x0001},
{0x000ED0, 0x0002},
{0x000EDA, 0x0001},
{0x000EDC, 0x0004},
{0x000EE0, 0x0001},
{0x000F00, 0x0004},
{0x000F01, 0x0040},
{0x000F04, 0x0020},
{0x000F13, 0x0040},
{0x000F14, 0x0020},
{0x000F15, 0x0040},
{0x000F18, 0x0010},
{0x000F1A, 0x0040},
{0x000F20, 0x0002},
{0x000F34, 0x0040},
{0x000F35, 0x0010},
{0x000F36, 0x0040},
{0x000F37, 0x0010},
{0x000F38, 0x0040},
{0x000F39, 0x0010},
{0x000F3A, 0x0020},
{0x000F3E, 0x0010},
{0x000F40, 0x0004},
{0x000F48, 0x0001},
{0x000F49, 0x0004},
{0x000F6D, 0x0001},
{0x000F71, 0x0010},
{0x000F85, 0x0020},
{0x000F86, 0x0010},
{0x000F88, 0x0004},
{0x000F8D, 0x0010},
{0x000F98, 0x0001},
{0x000F99, 0x0010},
{0x000FBD, 0x0001},
{0x000FBE, 0x0040},
{0x000FC6, 0x0010},
{0x000FC7, 0x0040},
{0x000FCD, 0x0001},
{0x000FCE, 0x0040},
{0x000FD0, 0x0020},
{0x000FD5, 0x0040},
{0x000FD9, 0x0020},
{0x000FDB, 0x0001},
{0x001000, 0x0004},
{0x00102B, 0x0010},
{0x00103F, 0x0004},
{0x001040, 0x0002},
{0x00104A, 0x0020},
{0x001050, 0x0004},
{0x001056, 0x0010},
{0x00105A, 0x0004},
{0x00105E, 0x0010},
{0x001061, 0x0004},
{0x001062, 0x0010},
{0x001065, 0x0004},
{0x001067, 0x0010},
{0x00106E, 0x0004},
{0x001071, 0x0010},
{0x001075, 0x0004},
{0x001082, 0x0010},
{0x00108E, 0x0004},
{0x00108F, 0x0010},
{0x001090, 0x0002},
{0x00109A, 0x0010},
{0x00109E, 0x0040},
{0x0010A0, 0x0004},
{0x0010C6, 0x0001},
{0x0010C7, 0x0004},
{0x0010C8, 0x0001},
{0x0010CD, 0x0004},
{0x0010CE, 0x0001},
{0x0010D0, 0x0004},
{0x0010FB, 0x0020},
{0x0010FC, 0x0004},
{0x001249, 0x0001},
{0x00124A, 0x0004},
{0x00124E, 0x0001},
{0x001250, 0x0004},
{0x001257, 0x0001},
{0x001258, 0x0004},
{0x001259, 0x0001},
{0x00125A, 0x0004},
{0x00125E, 0x0001},
{0x001260, 0x0004},
{0x001289, 0x0001},
{0x00128A, 0x0004},
{0x00128E, 0x0001},
{0x001290, 0x0004},
{0x0012B1, 0x0001},
{0x0012B2, 0x0004},
{0x0012B6, 0x0001},
{0x0012B8, 0x0004},
{0x0012BF, 0x0001},
{0x0012C0, 0x0004},
{0x0012C1, 0x0001},
{0x0012C2, 0x0004},
{0x0012C6, 0x0001},
{0x0012C8, 0x0004},
{0x0012D7, 0x0001},
{0x0012D8, 0x0004},
{0x001311, 0x0001},
{0x001312, 0x0004},
{0x001316, 0x0001},
{0x001318, 0x0004},
{0x00135B, 0x0001},
{0x00135D, 0x0010},
{0x001360, 0x0020},
{0x001369, 0x0002},
{0x00137D, 0x0001},
{0x001380, 0x0004},
{0x001390, 0x0040},
{0x00139A, 0x0001},
{0x0013A0, 0x0004},
{0x0013F6, 0x0001},
{0x0013F8, 0x0004},
{0x0013FE, 0x0001},
{0x001400, 0x0020},
{0x001401, 0x0004},
{0x00166D, 0x0040},
{0x00166E, 0x0020},
{0x00166F, 0x0004},
{0x001680, 0x0008},
{0x001681, 0x0004},
{0x00169B, 0x0020},
{0x00169D, 0x0001},
{0x0016A0, 0x0004},
{0x0016EB, 0x0020},
{0x0016EE, 0x0002},
{0x0016F1, 0x0004},
{0x0016F9, 0x0001},
{0x001700, 0x0004},
{0x001712, 0x0010},
{0x001716, 0x0001},
{0x00171F, 0x0004},
{0x001732, 0x0010},
{0x001735, 0x0020},
{0x001737, 0x0001},
{0x001740, 0x0004},
{0x001752, 0x0010},
{0x001754, 0x0001},
{0x001760, 0x0004},
{0x00176D, 0x0001},
{0x00176E, 0x0004},
{0x001771, 0x0001},
{0x001772, 0x0010},
{0x001774, 0x0001},
{0x001780, 0x0004},
{0x0017B4, 0x0010},
{0x0017D4, 0x0020},
{0x0017D7, 0x0004},
{0x0017D8, 0x0020},
{0x0017DB, 0x0040},
{0x0017DC, 0x0004},
{0x0017DD, 0x0010},
{0x0017DE, 0x0001},
{0x0017E0, 0x0002},
{0x0017EA, 0x0001},
{0x0017F0, 0x0002},
{0x0017FA, 0x0001},
{0x001800, 0x0020},
{0x00180B, 0x0010},
{0x00180E, 0x0080},
{0x00180F, 0x0010},
{0x001810, 0x0002},
{0x00181A, 0x0001},
{0x001820, 0x0004},
{0x001879, 0x0001},
{0x001880, 0x0004},
{0x001885, 0x0010},
{0x001887, 0x0004},
{0x0018A9, 0x0010},
{0x0018AA, 0x0004},
{0x0018AB, 0x0001},
{0x0018B0, 0x0004},
{0x0018F6, 0x0001},
{0x001900, 0x0004},
{0x00191F, 0x0001},
{0x001920, 0x0010},
{0x00192C, 0x0001},
{0x001930, 0x0010},
{0x00193C, 0x0001},
{0x001940, 0x0040},
{0x001941, 0x0001},
{0x001944, 0x0020},
{0x001946, 0x0002},
{0x001950, 0x0004},
{0x00196E, 0x0001},
{0x001970, 0x0004},
{0x001975, 0x0001},
{0x001980, 0x0004},
{0x0019AC, 0x0001},
{0x0019B0, 0x0004},
{0x0019CA, 0x0001},
{0x0019D0, 0x0002},
{0x0019DB, 0x0001},
{0x0019DE, 0x0040},
{0x001A00, 0x0004},
{0x001A17, 0x0010},
{0x001A1C, 0x0001},
{0x001A1E, 0x0020},
{0x001A20, 0x0004},
{0x001A55, 0x0010},
{0x001A5F, 0x0001},
{0x001A60, 0x0010},
{0x001A7D, 0x0001},
{0x001A7F, 0x0010},
{0x001A80, 0x0002},
{0x001A8A, 0x0001},
{0x001A90, 0x0002},
{0x001A9A, 0x0001},
{0x001AA0, 0x0020},
{0x001AA7, 0x0004},
{0x001AA8, 0x0020},
{0x001AAE, 0x0001},
{0x001AB0, 0x0010},
{0x001ACF, 0x0001},
{0x001B00, 0x0010},
{0x001B05, 0x0004},
{0x001B34, 0x0010},
{0x001B45, 0x0004},
{0x001B4D, 0x0001},
{0x001B50, 0x0002},
{0x001B5A, 0x0020},
{0x001B61, 0x0040},
{0x001B6B, 0x0010},
{0x001B74, 0x0040},
{0x001B7D, 0x0020},
{0x001B7F, 0x0001},
{0x001B80, 0x0010},
{0x001B83, 0x0004},
{0x001BA1, 0x0010},
{0x001BAE, 0x0004},
{0x001BB0, 0x0002},
{0x001BBA, 0x0004},
{0x001BE6, 0x0010},
{0x001BF4, 0x0001},
{0x001BFC, 0x0020},
{0x001C00, 0x0004},
{0x001C24, 0x0010},
{0x001C38, 0x0001},
{0x001C3B, 0x0020},
{0x001C40, 0x0002},
{0x001C4A, 0x0001},
{0x001C4D, 0x0004},
{0x001C50, 0x0002},
{0x001C5A, 0x0004},
{0x001C7E, 0x0020},
{0x001C80, 0x0004},
{0x001C89, 0x0001},
{0x001C90, 0x0004},
{0x001CBB, 0x0001},
{0x001CBD, 0x0004},
{0x001CC0, 0x0020},
{0x001CC8, 0x0001},
{0x001CD0, 0x0010},
{0x001CD3, 0x0020},
{0x001CD4, 0x0010},
{0x001CE9, 0x0004},
{0x001CED, 0x0010},
{0x001CEE, 0x0004},
{0x001CF4, 0x0010},
{0x001CF5, 0x0004},
{0x001CF7, 0x0010},
{0x001CFA, 0x0004},
{0x001CFB, 0x0001},
{0x001D00, 0x0004},
{0x001DC0, 0x0010},
{0x001E00, 0x0004},
{0x001F16, 0x0001},
{0x001F18, 0x0004},
{0x001F1E, 0x0001},
{0x001F20, 0x0004},
{0x001F46, 0x0001},
{0x001F48, 0x0004},
{0x001F4E, 0x0001},
{0x001F50, 0x0004},
{0x001F58, 0x0001},
{0x001F59, 0x0004},
{0x001F5A, 0x0001},
{0x001F5B, 0x0004},
{0x001F5C, 0x0001},
{0x001F5D, 0x0004},
{0x001F5E, 0x0001},
{0x001F5F, 0x0004},
{0x001F7E, 0x0001},
{0x001F80, 0x0004},
{0x001FB5, 0x0001},
{0x001FB6, 0x0004},
{0x001FBD, 0x0040},
{0x001FBE, 0x0004},
{0x001FBF, 0x0040},
{0x001FC2, 0x0004},
{0x001FC5, 0x0001},
{0x001FC6, 0x0004},
{0x001FCD, 0x0040},
{0x001FD0, 0x0004},
{0x001FD4, 0x0001},
{0x001FD6, 0x0004},
{0x001FDC, 0x0001},
{0x001FDD, 0x0040},
{0x001FE0, 0x0004},
{0x001FED, 0x0040},
{0x001FF0, 0x0001},
{0x001FF2, 0x0004},
{0x001FF5, 0x0001},
{0x001FF6, 0x0004},
{0x001FFD, 0x0040},
{0x001FFF, 0x0001},
{0x002000, 0x0008},
{0x00200B, 0x0080},
{0x002010, 0x0020},
{0x002028, 0x0008},
{0x00202A, 0x0080},
{0x00202F, 0x0008},
{0x002030, 0x0020},
{0x002044, 0x0040},
{0x002045, 0x0020},
{0x002052, 0x0040},
{0x002053, 0x0020},
{0x00205F, 0x0008},
{0x002060, 0x0080},
{0x002065, 0x0001},
{0x002066, 0x0080},
{0x002070, 0x0002},
{0x002071, 0x0004},
{0x002072, 0x0001},
{0x002074, 0x0002},
{0x00207A, 0x0040},
{0x00207D, 0x0020},
{0x00207F, 0x0004},
{0x002080, 0x0002},
{0x00208A, 0x0040},
{0x00208D, 0x0020},
{0x00208F, 0x0001},
{0x002090, 0x0004},
{0x00209D, 0x0001},
{0x0020A0, 0x0040},
{0x0020C1, 0x0001},
{0x0020D0, 0x0010},
{0x0020F1, 0x0001},
{0x002100, 0x0040},
{0x002102, 0x0004},
{0x002103, 0x0040},
{0x002107, 0x0004},
{0x002108, 0x0040},
{0x00210A, 0x0004},
{0x002114, 0x0040},
{0x002115, 0x0004},
{0x002116, 0x0040},
{0x002119, 0x0004},
{0x00211E, 0x0040},
{0x002124, 0x0004},
{0x002125, 0x0040},
{0x002126, 0x0004},
{0x002127, 0x0040},
{0x002128, 0x0004},
{0x002129, 0x0040},
{0x00212A, 0x0004},
{0x00212E, 0x0040},
{0x00212F, 0x0004},
{0x00213A, 0x0040},
{0x00213C, 0x0004},
{0x002140, 0x0040},
{0x002145, 0x0004},
{0x00214A, 0x0040},
{0x00214E, 0x0004},
{0x00214F, 0x0040},
{0x002150, 0x0002},
{0x002183, 0x0004},
{0x002185, 0x0002},
{0x00218A, 0x0040},
{0x00218C, 0x0001},
{0x002190, 0x0040},
{0x002308, 0x0020},
{0x00230C, 0x0040},
{0x002329, 0x0020},
{0x00232B, 0x0040},
{0x002427, 0x0001},
{0x002440, 0x0040},
{0x00244B, 0x0001},
{0x002460, 0x0002},
{0x00249C, 0x0040},
{0x0024EA, 0x0002},
{0x002500, 0x0040},
{0x002768, 0x0020},
{0x002776, 0x0002},
{0x002794, 0x0040},
{0x0027C5, 0x0020},
{0x0027C7, 0x0040},
{0x0027E6, 0x0020},
{0x0027F0, 0x0040},
{0x002983, 0x0020},
{0x002999, 0x0040},
{0x0029D8, 0x0020},
{0x0029DC, 0x0040},
{0x0029FC, 0x0020},
{0x0029FE, 0x0040},
{0x002B74, 0x0001},
{0x002B76, 0x0040},
{0x002B96, 0x0001},
{0x002B97, 0x0040},
{0x002C00, 0x0004},
{0x002CE5, 0x0040},
{0x002CEB, 0x0004},
{0x002CEF, 0x0010},
{0x002CF2, 0x0004},
{0x002CF4, 0x0001},
{0x002CF9, 0x0020},
{0x002CFD, 0x0002},
{0x002CFE, 0x0020},
{0x002D00, 0x0004},
{0x002D26, 0x0001},
{0x002D27, 0x0004},
{0x002D28, 0x0001},
{0x002D2D, 0x0004},
{0x002D2E, 0x0001},
{0x002D30, 0x0004},
{0x002D68, 0x0001},
{0x002D6F, 0x0004},
{0x002D70, 0x0020},
{0x002D71, 0x0001},
{0x002D7F, 0x0010},
{0x002D80, 0x0004},
{0x002D97, 0x0001},
{0x002DA0, 0x0004},
{0x002DA7, 0x0001},
{0x002DA8, 0x0004},
{0x002DAF, 0x0001},
{0x002DB0, 0x0004},
{0x002DB7, 0x0001},
{0x002DB8, 0x0004},
{0x002DBF, 0x0001},
{0x002DC0, 0x0004},
{0x002DC7, 0x0001},
{0x002DC8, 0x0004},
{0x002DCF, 0x0001},
{0x002DD0, 0x0004},
{0x002DD7, 0x0001},
{0x002DD8, 0x0004},
{0x002DDF, 0x0001},
{0x002DE0, 0x0010},
{0x002E00, 0x0020},
{0x002E2F, 0x0004},
{0x002E30, 0x0020},
{0x002E50, 0x0040},
{0x002E52, 0x0020},
{0x002E5E, 0x0001},
{0x002E80, 0x0040},
{0x002E9A, 0x0001},
{0x002E9B, 0x0040},
{0x002EF4, 0x0001},
{0x002F00, 0x0040},
{0x002FD6, 0x0001},
{0x002FF0, 0x0040},
{0x003000, 0x0008},
{0x003001, 0x0020},
{0x003004, 0x0040},
{0x003005, 0x0004},
{0x003007, 0x0002},
{0x003008, 0x0020},
{0x003012, 0x0040},
{0x003014, 0x0020},
{0x003020, 0x0040},
{0x003021, 0x0002},
{0x00302A, 0x0010},
{0x003030, 0x0020},
{0x003031, 0x0004},
{0x003036, 0x0040},
{0x003038, 0x0002},
{0x00303B, 0x0004},
{0x00303D, 0x0020},
{0x00303E, 0x0040},
{0x003040, 0x0001},
{0x003041, 0x0004},
{0x003097, 0x0001},
{0x003099, 0x0010},
{0x00309B, 0x0040},
{0x00309D, 0x0004},
{0x0030A0, 0x0020},
{0x0030A1, 0x0004},
{0x0030FB, 0x0020},
{0x0030FC, 0x0004},
{0x003100, 0x0001},
{0x003105, 0x0004},
{0x003130, 0x0001},
{0x003131, 0x0004},
{0x00318F, 0x0001},
{0x003190, 0x0040},
{0x003192, 0x0002},
{0x003196, 0x0040},
{0x0031A0, 0x0004},
{0x0031C0, 0x0040},
{0x0031E4, 0x0001},
{0x0031EF, 0x0040},
{0x0031F0, 0x0004},
{0x003200, 0x0040},
{0x00321F, 0x0001},
{0x003220, 0x0002},
{0x00322A, 0x0040},
{0x003248, 0x0002},
{0x003250, 0x0040},
{0x003251, 0x0002},
{0x003260, 0x0040},
{0x003280, 0x0002},
{0x00328A, 0x0040},
{0x0032B1, 0x0002},
{0x0032C0, 0x0040},
{0x003400, 0x0004},
{0x004DC0, 0x0040},
{0x004E00, 0x0004},
{0x00A48D, 0x0001},
{0x00A490, 0x0040},
{0x00A4C7, 0x0001},
{0x00A4D0, 0x0004},
{0x00A4FE, 0x0020},
{0x00A500, 0x0004},
{0x00A60D, 0x0020},
{0x00A610, 0x0004},
{0x00A620, 0x0002},
{0x00A62A, 0x0004},
{0x00A62C, 0x0001},
{0x00A640, 0x0004},
{0x00A66F, 0x0010},
{0x00A673, 0x0020},
{0x00A674, 0x0010},
{0x00A67E, 0x0020},
{0x00A67F, 0x0004},
{0x00A69E, 0x0010},
{0x00A6A0, 0x0004},
{0x00A6E6, 0x0002},
{0x00A6F0, 0x0010},
{0x00A6F2, 0x0020},
{0x00A6F8, 0x0001},
{0x00A700, 0x0040},
{0x00A717, 0x0004},
{0x00A720, 0x0040},
{0x00A722, 0x0004},
{0x00A789, 0x0040},
{0x00A78B, 0x0004},
{0x00A7CB, 0x0001},
{0x00A7D0, 0x0004},
{0x00A7D2, 0x0001},
{0x00A7D3, 0x0004},
{0x00A7D4, 0x0001},
{0x00A7D5, 0x0004},
{0x00A7DA, 0x0001},
{0x00A7F2, 0x0004},
{0x00A802, 0x0010},
{0x00A803, 0x0004},
{0x00A806, 0x0010},
{0x00A807, 0x0004},
{0x00A80B, 0x0010},
{0x00A80C, 0x0004},
{0x00A823, 0x0010},
{0x00A828, 0x0040},
{0x00A82C, 0x0010},
{0x00A82D, 0x0001},
{0x00A830, 0x0002},
{0x00A836, 0x0040},
{0x00A83A, 0x0001},
{0x00A840, 0x0004},
{0x00A874, 0x0020},
{0x00A878, 0x0001},
{0x00A880, 0x0010},
{0x00A882, 0x0004},
{0x00A8B4, 0x0010},
{0x00A8C6, 0x0001},
{0x00A8CE, 0x0020},
{0x00A8D0, 0x0002},
{0x00A8DA, 0x0001},
{0x00A8E0, 0x0010},
{0x00A8F2, 0x0004},
{0x00A8F8, 0x0020},
{0x00A8FB, 0x0004},
{0x00A8FC, 0x0020},
{0x00A8FD, 0x0004},
{0x00A8FF, 0x0010},
{0x00A900, 0x0002},
{0x00A90A, 0x0004},
{0x00A926, 0x0010},
{0x00A92E, 0x0020},
{0x00A930, 0x0004},
{0x00A947, 0x0010},
{0x00A954, 0x0001},
{0x00A95F, 0x0020},
{0x00A960, 0x0004},
{0x00A97D, 0x0001},
{0x00A980, 0x0010},
{0x00A984, 0x0004},
{0x00A9B3, 0x0010},
{0x00A9C1, 0x0020},
{0x00A9CE, 0x0001},
{0x00A9CF, 0x0004},
{0x00A9D0, 0x0002},
{0x00A9DA, 0x0001},
{0x00A9DE, 0x0020},
{0x00A9E0, 0x0004},
{0x00A9E5, 0x0010},
{0x00A9E6, 0x0004},
{0x00A9F0, 0x0002},
{0x00A9FA, 0x0004},
{0x00A9FF, 0x0001},
{0x00AA00, 0x0004},
{0x00AA29, 0x0010},
{0x00AA37, 0x0001},
{0x00AA40, 0x0004},
{0x00AA43, 0x0010},
{0x00AA44, 0x0004},
{0x00AA4C, 0x0010},
{0x00AA4E, 0x0001},
{0x00AA50, 0x0002},
{0x00AA5A, 0x0001},
{0x00AA5C, 0x0020},
{0x00AA60, 0x0004},
{0x00AA77, 0x0040},
{0x00AA7A, 0x0004},
{0x00AA7B, 0x0010},
{0x00AA7E, 0x0004},
{0x00AAB0, 0x0010},
{0x00AAB1, 0x0004},
{0x00AAB2, 0x0010},
{0x00AAB5, 0x0004},
{0x00AAB7, 0x0010},
{0x00AAB9, 0x0004},
{0x00AABE, 0x0010},
{0x00AAC0, 0x0004},
{0x00AAC1, 0x0010},
{0x00AAC2, 0x0004},
{0x00AAC3, 0x0001},
{0x00AADB, 0x0004},
{0x00AADE, 0x0020},
{0x00AAE0, 0x0004},
{0x00AAEB, 0x0010},
{0x00AAF0, 0x0020},
{0x00AAF2, 0x0004},
{0x00AAF5, 0x0010},
{0x00AAF7, 0x0001},
{0x00AB01, 0x0004},
{0x00AB07, 0x0001},
{0x00AB09, 0x0004},
{0x00AB0F, 0x0001},
{0x00AB11, 0x0004},
{0x00AB17, 0x0001},
{0x00AB20, 0x0004},
{0x00AB27, 0x0001},
{0x00AB28, 0x0004},
{0x00AB2F, 0x0001},
{0x00AB30, 0x0004},
{0x00AB5B, 0x0040},
{0x00AB5C, 0x0004},
{0x00AB6A, 0x0040},
{0x00AB6C, 0x0001},
{0x00AB70, 0x0004},
{0x00ABE3, 0x0010},
{0x00ABEB, 0x0020},
{0x00ABEC, 0x0010},
{0x00ABEE, 0x0001},
{0x00ABF0, 0x0002},
{0x00ABFA, 0x0001},
{0x00AC00, 0x0004},
{0x00D7A4, 0x0001},
{0x00D7B0, 0x0004},
{0x00D7C7, 0x0001},
{0x00D7CB, 0x0004},
{0x00D7FC, 0x0001},
{0x00D800, 0x0080},
{0x00F900, 0x0004},
{0x00FA6E, 0x0001},
{0x00FA70, 0x0004},
{0x00FADA, 0x0001},
{0x00FB00, 0x0004},
{0x00FB07, 0x0001},
{0x00FB13, 0x0004},
{0x00FB18, 0x0001},
{0x00FB1D, 0x0004},
{0x00FB1E, 0x0010},
{0x00FB1F, 0x0004},
{0x00FB29, 0x0040},
{0x00FB2A, 0x0004},
{0x00FB37, 0x0001},
{0x00FB38, 0x0004},
{0x00FB3D, 0x0001},
{0x00FB3E, 0x0004},
{0x00FB3F, 0x0001},
{0x00FB40, 0x0004},
{0x00FB42, 0x0001},
{0x00FB43, 0x0004},
{0x00FB45, 0x0001},
{0x00FB46, 0x0004},
{0x00FBB2, 0x0040},
{0x00FBC3, 0x0001},
{0x00FBD3, 0x0004},
{0x00FD3E, 0x0020},
{0x00FD40, 0x0040},
{0x00FD50, 0x0004},
{0x00FD90, 0x0001},
{0x00FD92, 0x0004},
{0x00FDC8, 0x0001},
{0x00FDCF, 0x0040},
{0x00FDD0, 0x0001},
{0x00FDF0, 0x0004},
{0x00FDFC, 0x0040},
{0x00FE00, 0x0010},
{0x00FE10, 0x0020},
{0x00FE1A, 0x0001},
{0x00FE20, 0x0010},
{0x00FE30, 0x0020},
{0x00FE53, 0x0001},
{0x00FE54, 0x0020},
{0x00FE62, 0x0040},
{0x00FE63, 0x0020},
{0x00FE64, 0x0040},
{0x00FE67, 0x0001},
{0x00FE68, 0x0020},
{0x00FE69, 0x0040},
{0x00FE6A, 0x0020},
{0x00FE6C, 0x0001},
{0x00FE70, 0x0004},
{0x00FE75, 0x0001},
{0x00FE76, 0x0004},
{0x00FEFD, 0x0001},
{0x00FEFF, 0x0080},
{0x00FF00, 0x0001},
{0x00FF01, 0x0020},
{0x00FF04, 0x0040},
{0x00FF05, 0x0020},
{0x00FF0B, 0x0040},
{0x00FF0C, 0x0020},
{0x00FF10, 0x0002},
{0x00FF1A, 0x0020},
{0x00FF1C, 0x0040},
{0x00FF1F, 0x0020},
{0x00FF21, 0x0004},
{0x00FF3B, 0x0020},
{0x00FF3E, 0x0040},
{0x00FF3F, 0x0020},
{0x00FF40, 0x0040},
{0x00FF41, 0x0004},
{0x00FF5B, 0x0020},
{0x00FF5C, 0x0040},
{0x00FF5D, 0x0020},
{0x00FF5E, 0x0040},
{0x00FF5F, 0x0020},
{0x00FF66, 0x0004},
{0x00FFBF, 0x0001},
{0x00FFC2, 0x0004},
{0x00FFC8, 0x0001},
{0x00FFCA, 0x0004},
{0x00FFD0, 0x0001},
{0x00FFD2, 0x0004},
{0x00FFD8, 0x0001},
{0x00FFDA, 0x0004},
{0x00FFDD, 0x0001},
{0x00FFE0, 0x0040},
{0x00FFE7, 0x0001},
{0x00FFE8, 0x0040},
{0x00FFEF, 0x0001},
{0x00FFF9, 0x0080},
{0x00FFFC, 0x0040},
{0x00FFFE, 0x0001},
{0x010000, 0x0004},
{0x01000C, 0x0001},
{0x01000D, 0x0004},
{0x010027, 0x0001},
{0x010028, 0x0004},
{0x01003B, 0x0001},
{0x01003C, 0x0004},
{0x01003E, 0x0001},
{0x01003F, 0x0004},
{0x01004E, 0x0001},
{0x010050, 0x0004},
{0x01005E, 0x0001},
{0x010080, 0x0004},
{0x0100FB, 0x0001},
{0x010100, 0x0020},
{0x010103, 0x0001},
{0x010107, 0x0002},
{0x010134, 0x0001},
{0x010137, 0x0040},
{0x010140, 0x0002},
{0x010179, 0x0040},
{0x01018A, 0x0002},
{0x01018C, 0x0040},
{0x01018F, 0x0001},
{0x010190, 0x0040},
{0x01019D, 0x0001},
{0x0101A0, 0x0040},
{0x0101A1, 0x0001},
{0x0101D0, 0x0040},
{0x0101FD, 0x0010},
{0x0101FE, 0x0001},
{0x010280, 0x0004},
{0x01029D, 0x0001},
{0x0102A0, 0x0004},
{0x0102D1, 0x0001},
{0x0102E0, 0x0010},
{0x0102E1, 0x0002},
{0x0102FC, 0x0001},
{0x010300, 0x0004},
{0x010320, 0x0002},
{0x010324, 0x0001},
{0x01032D, 0x0004},
{0x010341, 0x0002},
{0x010342, 0x0004},
{0x01034A, 0x0002},
{0x01034B, 0x0001},
{0x010350, 0x0004},
{0x010376, 0x0010},
{0x01037B, 0x0001},
{0x010380, 0x0004},
{0x01039E, 0x0001},
{0x01039F, 0x0020},
{0x0103A0, 0x0004},
{0x0103C4, 0x0001},
{0x0103C8, 0x0004},
{0x0103D0, 0x0020},
{0x0103D1, 0x0002},
{0x0103D6, 0x0001},
{0x010400, 0x0004},
{0x01049E, 0x0001},
{0x0104A0, 0x0002},
{0x0104AA, 0x0001},
{0x0104B0, 0x0004},
{0x0104D4, 0x0001},
{0x0104D8, 0x0004},
{0x0104FC, 0x0001},
{0x010500, 0x0004},
{0x010528, 0x0001},
{0x010530, 0x0004},
{0x010564, 0x0001},
{0x01056F, 0x0020},
{0x010570, 0x0004},
{0x01057B, 0x0001},
{0x01057C, 0x0004},
{0x01058B, 0x0001},
{0x01058C, 0x0004},
{0x010593, 0x0001},
{0x010594, 0x0004},
{0x010596, 0x0001},
{0x010597, 0x0004},
{0x0105A2, 0x0001},
{0x0105A3, 0x0004},
{0x0105B2, 0x0001},
{0x0105B3, 0x0004},
{0x0105BA, 0x0001},
{0x0105BB, 0x0004},
{0x0105BD, 0x0001},
{0x010600, 0x0004},
{0x010737, 0x0001},
{0x010740, 0x0004},
{0x010756, 0x0001},
{0x010760, 0x0004},
{0x010768, 0x0001},
{0x010780, 0x0004},
{0x010786, 0x0001},
{0x010787, 0x0004},
{0x0107B1, 0x0001},
{0x0107B2, 0x0004},
{0x0107BB, 0x0001},
{0x010800, 0x0004},
{0x010806, 0x0001},
{0x010808, 0x0004},
{0x010809, 0x0001},
{0x01080A, 0x0004},
{0x010836, 0x0001},
{0x010837, 0x0004},
{0x010839, 0x0001},
{0x01083C, 0x0004},
{0x01083D, 0x0001},
{0x01083F, 0x0004},
{0x010856, 0x0001},
{0x010857, 0x0020},
{0x010858, 0x0002},
{0x010860, 0x0004},
{0x010877, 0x0040},
{0x010879, 0x0002},
{0x010880, 0x0004},
{0x01089F, 0x0001},
{0x0108A7, 0x0002},
{0x0108B0, 0x0001},
{0x0108E0, 0x0004},
{0x0108F3, 0x0001},
{0x0108F4, 0x0004},
{0x0108F6, 0x0001},
{0x0108FB, 0x0002},
{0x010900, 0x0004},
{0x010916, 0x0002},
{0x01091C, 0x0001},
{0x01091F, 0x0020},
{0x010920, 0x0004},
{0x01093A, 0x0001},
{0x01093F, 0x0020},
{0x010940, 0x0001},
{0x010980, 0x0004},
{0x0109B8, 0x0001},
{0x0109BC, 0x0002},
{0x0109BE, 0x0004},
{0x0109C0, 0x0002},
{0x0109D0, 0x0001},
{0x0109D2, 0x0002},
{0x010A00, 0x0004},
{0x010A01, 0x0010},
{0x010A04, 0x0001},
{0x010A05, 0x0010},
{0x010A07, 0x0001},
{0x010A0C, 0x0010},
{0x010A10, 0x0004},
{0x010A14, 0x0001},
{0x010A15, 0x0004},
{0x010A18, 0x0001},
{0x010A19, 0x0004},
{0x010A36, 0x0001},
{0x010A38, 0x0010},
{0x010A3B, 0x0001},
{0x010A3F, 0x0010},
{0x010A40, 0x0002},
{0x010A49, 0x0001},
{0x010A50, 0x0020},
{0x010A59, 0x0001},
{0x010A60, 0x0004},
{0x010A7D, 0x0002},
{0x010A7F, 0x0020},
{0x010A80, 0x0004},
{0x010A9D, 0x0002},
{0x010AA0, 0x0001},
{0x010AC0, 0x0004},
{0x010AC8, 0x0040},
{0x010AC9, 0x0004},
{0x010AE5, 0x0010},
{0x010AE7, 0x0001},
{0x010AEB, 0x0002},
{0x010AF0, 0x0020},
{0x010AF7, 0x0001},
{0x010B00, 0x0004},
{0x010B36, 0x0001},
{0x010B39, 0x0020},
{0x010B40, 0x0004},
{0x010B56, 0x0001},
{0x010B58, 0x0002},
{0x010B60, 0x0004},
{0x010B73, 0x0001},
{0x010B78, 0x0002},
{0x010B80, 0x0004},
{0x010B92, 0x0001},
{0x010B99, 0x0020},
{0x010B9D, 0x0001},
{0x010BA9, 0x0002},
{0x010BB0, 0x0001},
{0x010C00, 0x0004},
{0x010C49, 0x0001},
{0x010C80, 0x0004},
{0x010CB3, 0x0001},
{0x010CC0, 0x0004},
{0x010CF3, 0x0001},
{0x010CFA, 0x0002},
{0x010D00, 0x0004},
{0x010D24, 0x0010},
{0x010D28, 0x0001},
{0x010D30, 0x0002},
{0x010D3A, 0x0001},
{0x010E60, 0x0002},
{0x010E7F, 0x0001},
{0x010E80, 0x0004},
{0x010EAA, 0x0001},
{0x010EAB, 0x0010},
{0x010EAD, 0x0020},
{0x010EAE, 0x0001},
{0x010EB0, 0x0004},
{0x010EB2, 0x0001},
{0x010EFD, 0x0010},
{0x010F00, 0x0004},
{0x010F1D, 0x0002},
{0x010F27, 0x0004},
{0x010F28, 0x0001},
{0x010F30, 0x0004},
{0x010F46, 0x0010},
{0x010F51, 0x0002},
{0x010F55, 0x0020},
{0x010F5A, 0x0001},
{0x010F70, 0x0004},
{0x010F82, 0x0010},
{0x010F86, 0x0020},
{0x010F8A, 0x0001},
{0x010FB0, 0x0004},
{0x010FC5, 0x0002},
{0x010FCC, 0x0001},
{0x010FE0, 0x0004},
{0x010FF7, 0x0001},
{0x011000, 0x0010},
{0x011003, 0x0004},
{0x011038, 0x0010},
{0x011047, 0x0020},
{0x01104E, 0x0001},
{0x011052, 0x0002},
{0x011070, 0x0010},
{0x011071, 0x0004},
{0x011073, 0x0010},
{0x011075, 0x0004},
{0x011076, 0x0001},
{0x01107F, 0x0010},
{0x011083, 0x0004},
{0x0110B0, 0x0010},
{0x0110BB, 0x0020},
{0x0110BD, 0x0080},
{0x0110BE, 0x0020},
{0x0110C2, 0x0010},
{0x0110C3, 0x0001},
{0x0110CD, 0x0080},
{0x0110CE, 0x0001},
{0x0110D0, 0x0004},
{0x0110E9, 0x0001},
{0x0110F0, 0x0002},
{0x0110FA, 0x0001},
{0x011100, 0x0010},
{0x011103, 0x0004},
{0x011127, 0x0010},
{0x011135, 0x0001},
{0x011136, 0x0002},
{0x011140, 0x0020},
{0x011144, 0x0004},
{0x011145, 0x0010},
{0x011147, 0x0004},
{0x011148, 0x0001},
{0x011150, 0x0004},
{0x011173, 0x0010},
{0x011174, 0x0020},
{0x011176, 0x0004},
{0x011177, 0x0001},
{0x011180, 0x0010},
{0x011183, 0x0004},
{0x0111B3, 0x0010},
{0x0111C1, 0x0004},
{0x0111C5, 0x0020},
{0x0111C9, 0x0010},
{0x0111CD, 0x0020},
{0x0111CE, 0x0010},
{0x0111D0, 0x0002},
{0x0111DA, 0x0004},
{0x0111DB, 0x0020},
{0x0111DC, 0x0004},
{0x0111DD, 0x0020},
{0x0111E0, 0x0001},
{0x0111E1, 0x0002},
{0x0111F5, 0x0001},
{0x011200, 0x0004},
{0x011212, 0x0001},
{0x011213, 0x0004},
{0x01122C, 0x0010},
{0x011238, 0x0020},
{0x01123E, 0x0010},
{0x01123F, 0x0004},
{0x011241, 0x0010},
{0x011242, 0x0001},
{0x011280, 0x0004},
{0x011287, 0x0001},
{0x011288, 0x0004},
{0x011289, 0x0001},
{0x01128A, 0x0004},
{0x01128E, 0x0001},
{0x01128F, 0x0004},
{0x01129E, 0x0001},
{0x01129F, 0x0004},
{0x0112A9, 0x0020},
{0x0112AA, 0x0001},
{0x0112B0, 0x0004},
{0x0112DF, 0x0010},
{0x0112EB, 0x0001},
{0x0112F0, 0x0002},
{0x0112FA, 0x0001},
{0x011300, 0x0010},
{0x011304, 0x0001},
{0x011305, 0x0004},
{0x01130D, 0x0001},
{0x01130F, 0x0004},
{0x011311, 0x0001},
{0x011313, 0x0004},
{0x011329, 0x0001},
{0x01132A, 0x0004},
{0x011331, 0x0001},
{0x011332, 0x0004},
{0x011334, 0x0001},
{0x011335, 0x0004},
{0x01133A, 0x0001},
{0x01133B, 0x0010},
{0x01133D, 0x0004},
{0x01133E, 0x0010},
{0x011345, 0x0001},
{0x011347, 0x0010},
{0x011349, 0x0001},
{0x01134B, 0x0010},
{0x01134E, 0x0001},
{0x011350, 0x0004},
{0x011351, 0x0001},
{0x011357, 0x0010},
{0x011358, 0x0001},
{0x01135D, 0x0004},
{0x011362, 0x0010},
{0x011364, 0x0001},
{0x011366, 0x0010},
{0x01136D, 0x0001},
{0x011370, 0x0010},
{0x011375, 0x0001},
{0x011400, 0x0004},
{0x011435, 0x0010},
{0x011447, 0x0004},
{0x01144B, 0x0020},
{0x011450, 0x0002},
{0x01145A, 0x0020},
{0x01145C, 0x0001},
{0x01145D, 0x0020},
{0x01145E, 0x0010},
{0x01145F, 0x0004},
{0x011462, 0x0001},
{0x011480, 0x0004},
{0x0114B0, 0x0010},
{0x0114C4, 0x0004},
{0x0114C6, 0x0020},
{0x0114C7, 0x0004},
{0x0114C8, 0x0001},
{0x0114D0, 0x0002},
{0x0114DA, 0x0001},
{0x011580, 0x0004},
{0x0115AF, 0x0010},
{0x0115B6, 0x0001},
{0x0115B8, 0x0010},
{0x0115C1, 0x0020},
{0x0115D8, 0x0004},
{0x0115DC, 0x0010},
{0x0115DE, 0x0001},
{0x011600, 0x0004},
{0x011630, 0x0010},
{0x011641, 0x0020},
{0x011644, 0x0004},
{0x011645, 0x0001},
{0x011650, 0x0002},
{0x01165A, 0x0001},
{0x011660, 0x0020},
{0x01166D, 0x0001},
{0x011680, 0x0004},
{0x0116AB, 0x0010},
{0x0116B8, 0x0004},
{0x0116B9, 0x0020},
{0x0116BA, 0x0001},
{0x0116C0, 0x0002},
{0x0116CA, 0x0001},
{0x011700, 0x0004},
{0x01171B, 0x0001},
{0x01171D, 0x0010},
{0x01172C, 0x0001},
{0x011730, 0x0002},
{0x01173C, 0x0020},
{0x01173F, 0x0040},
{0x011740, 0x0004},
{0x011747, 0x0001},
{0x011800, 0x0004},
{0x01182C, 0x0010},
{0x01183B, 0x0020},
{0x01183C, 0x0001},
{0x0118A0, 0x0004},
{0x0118E0, 0x0002},
{0x0118F3, 0x0001},
{0x0118FF, 0x0004},
{0x011907, 0x0001},
{0x011909, 0x0004},
{0x01190A, 0x0001},
{0x01190C, 0x0004},
{0x011914, 0x0001},
{0x011915, 0x0004},
{0x011917, 0x0001},
{0x011918, 0x0004},
{0x011930, 0x0010},
{0x011936, 0x0001},
{0x011937, 0x0010},
{0x011939, 0x0001},
{0x01193B, 0x0010},
{0x01193F, 0x0004},
{0x011940, 0x0010},
{0x011941, 0x0004},
{0x011942, 0x0010},
{0x011944, 0x0020},
{0x011947, 0x0001},
{0x011950, 0x0002},
{0x01195A, 0x0001},
{0x0119A0, 0x0004},
{0x0119A8, 0x0001},
{0x0119AA, 0x0004},
{0x0119D1, 0x0010},
{0x0119D8, 0x0001},
{0x0119DA, 0x0010},
{0x0119E1, 0x0004},
{0x0119E2, 0x0020},
{0x0119E3, 0x0004},
{0x0119E4, 0x0010},
{0x0119E5, 0x0001},
{0x011A00, 0x0004},
{0x011A01, 0x0010},
{0x011A0B, 0x0004},
{0x011A33, 0x0010},
{0x011A3A, 0x0004},
{0x011A3B, 0x0010},
{0x011A3F, 0x0020},
{0x011A47, 0x0010},
{0x011A48, 0x0001},
{0x011A50, 0x0004},
{0x011A51, 0x0010},
{0x011A5C, 0x0004},
{0x011A8A, 0x0010},
{0x011A9A, 0x0020},
{0x011A9D, 0x0004},
{0x011A9E, 0x0020},
{0x011AA3, 0x0001},
{0x011AB0, 0x0004},
{0x011AF9, 0x0001},
{0x011B00, 0x0020},
{0x011B0A, 0x0001},
{0x011C00, 0x0004},
{0x011C09, 0x0001},
{0x011C0A, 0x0004},
{0x011C2F, 0x0010},
{0x011C37, 0x0001},
{0x011C38, 0x0010},
{0x011C40, 0x0004},
{0x011C41, 0x0020},
{0x011C46, 0x0001},
{0x011C50, 0x0002},
{0x011C6D, 0x0001},
{0x011C70, 0x0020},
{0x011C72, 0x0004},
{0x011C90, 0x0001},
{0x011C92, 0x0010},
{0x011CA8, 0x0001},
{0x011CA9, 0x0010},
{0x011CB7, 0x0001},
{0x011D00, 0x0004},
{0x011D07, 0x0001},
{0x011D08, 0x0004},
{0x011D0A, 0x0001},
{0x011D0B, 0x0004},
{0x011D31, 0x0010},
{0x011D37, 0x0001},
{0x011D3A, 0x0010},
{0x011D3B, 0x0001},
{0x011D3C, 0x0010},
{0x011D3E, 0x0001},
{0x011D3F, 0x0010},
{0x011D46, 0x0004},
{0x011D47, 0x0010},
{0x011D48, 0x0001},
{0x011D50, 0x0002},
{0x011D5A, 0x0001},
{0x011D60, 0x0004},
{0x011D66, 0x0001},
{0x011D67, 0x0004},
{0x011D69, 0x0001},
{0x011D6A, 0x0004},
{0x011D8A, 0x0010},
{0x011D8F, 0x0001},
{0x011D90, 0x0010},
{0x011D92, 0x0001},
{0x011D93, 0x0010},
{0x011D98, 0x0004},
{0x011D99, 0x0001},
{0x011DA0, 0x0002},
{0x011DAA, 0x0001},
{0x011EE0, 0x0004},
{0x011EF3, 0x0010},
{0x011EF7, 0x0020},
{0x011EF9, 0x0001},
{0x011F00, 0x0010},
{0x011F02, 0x0004},
{0x011F03, 0x0010},
{0x011F04, 0x0004},
{0x011F11, 0x0001},
{0x011F12, 0x0004},
{0x011F34, 0x0010},
{0x011F3B, 0x0001},
{0x011F3E, 0x0010},
{0x011F43, 0x0020},
{0x011F50, 0x0002},
{0x011F5A, 0x0001},
{0x011FB0, 0x0004},
{0x011FB1, 0x0001},
{0x011FC0, 0x0002},
{0x011FD5, 0x0040},
{0x011FF2, 0x0001},
{0x011FFF, 0x0020},
{0x012000, 0x0004},
{0x01239A, 0x0001},
{0x012400, 0x0002},
{0x01246F, 0x0001},
{0x012470, 0x0020},
{0x012475, 0x0001},
{0x012480, 0x0004},
{0x012544, 0x0001},
{0x012F90, 0x0004},
{0x012FF1, 0x0020},
{0x012FF3, 0x0001},
{0x013000, 0x0004},
{0x013430, 0x0080},
{0x013440, 0x0010},
{0x013441, 0x0004},
{0x013447, 0x0010},
{0x013456, 0x0001},
{0x014400, 0x0004},
{0x014647, 0x0001},
{0x016800, 0x0004},
{0x016A39, 0x0001},
{0x016A40, 0x0004},
{0x016A5F, 0x0001},
{0x016A60, 0x0002},
{0x016A6A, 0x0001},
{0x016A6E, 0x0020},
{0x016A70, 0x0004},
{0x016ABF, 0x0001},
{0x016AC0, 0x0002},
{0x016ACA, 0x0001},
{0x016AD0, 0x0004},
{0x016AEE, 0x0001},
{0x016AF0, 0x0010},
{0x016AF5, 0x0020},
{0x016AF6, 0x0001},
{0x016B00, 0x0004},
{0x016B30, 0x0010},
{0x016B37, 0x0020},
{0x016B3C, 0x0040},
{0x016B40, 0x0004},
{0x016B44, 0x0020},
{0x016B45, 0x0040},
{0x016B46, 0x0001},
{0x016B50, 0x0002},
{0x016B5A, 0x0001},
{0x016B5B, 0x0002},
{0x016B62, 0x0001},
{0x016B63, 0x0004},
{0x016B78, 0x0001},
{0x016B7D, 0x0004},
{0x016B90, 0x0001},
{0x016E40, 0x0004},
{0x016E80, 0x0002},
{0x016E97, 0x0020},
{0x016E9B, 0x0001},
{0x016F00, 0x0004},
{0x016F4B, 0x0001},
{0x016F4F, 0x0010},
{0x016F50, 0x0004},
{0x016F51, 0x0010},
{0x016F88, 0x0001},
{0x016F8F, 0x0010},
{0x016F93, 0x0004},
{0x016FA0, 0x0001},
{0x016FE0, 0x0004},
{0x016FE2, 0x0020},
{0x016FE3, 0x0004},
{0x016FE4, 0x0010},
{0x016FE5, 0x0001},
{0x016FF0, 0x0010},
{0x016FF2, 0x0001},
{0x017000, 0x0004},
{0x0187F8, 0x0001},
{0x018800, 0x0004},
{0x018CD6, 0x0001},
{0x018D00, 0x0004},
{0x018D09, 0x0001},
{0x01AFF0, 0x0004},
{0x01AFF4, 0x0001},
{0x01AFF5, 0x0004},
{0x01AFFC, 0x0001},
{0x01AFFD, 0x0004},
{0x01AFFF, 0x0001},
{0x01B000, 0x0004},
{0x01B123, 0x0001},
{0x01B132, 0x0004},
{0x01B133, 0x0001},
{0x01B150, 0x0004},
{0x01B153, 0x0001},
{0x01B155, 0x0004},
{0x01B156, 0x0001},
{0x01B164, 0x0004},
{0x01B168, 0x0001},
{0x01B170, 0x0004},
{0x01B2FC, 0x0001},
{0x01BC00, 0x0004},
{0x01BC6B, 0x0001},
{0x01BC70, 0x0004},
{0x01BC7D, 0x0001},
{0x01BC80, 0x0004},
{0x01BC89, 0x0001},
{0x01BC90, 0x0004},
{0x01BC9A, 0x0001},
{0x01BC9C, 0x0040},
{0x01BC9D, 0x0010},
{0x01BC9F, 0x0020},
{0x01BCA0, 0x0080},
{0x01BCA4, 0x0001},
{0x01CF00, 0x0010},
{0x01CF2E, 0x0001},
{0x01CF30, 0x0010},
{0x01CF47, 0x0001},
{0x01CF50, 0x0040},
{0x01CFC4, 0x0001},
{0x01D000, 0x0040},
{0x01D0F6, 0x0001},
{0x01D100, 0x0040},
{0x01D127, 0x0001},
{0x01D129, 0x0040},
{0x01D165, 0x0010},
{0x01D16A, 0x0040},
{0x01D16D, 0x0010},
{0x01D173, 0x0080},
{0x01D17B, 0x0010},
{0x01D183, 0x0040},
{0x01D185, 0x0010},
{0x01D18C, 0x0040},
{0x01D1AA, 0x0010},
{0x01D1AE, 0x0040},
{0x01D1EB, 0x0001},
{0x01D200, 0x0040},
{0x01D242, 0x0010},
{0x01D245, 0x0040},
{0x01D246, 0x0001},
{0x01D2C0, 0x0002},
{0x01D2D4, 0x0001},
{0x01D2E0, 0x0002},
{0x01D2F4, 0x0001},
{0x01D300, 0x0040},
{0x01D357, 0x0001},
{0x01D360, 0x0002},
{0x01D379, 0x0001},
{0x01D400, 0x0004},
{0x01D455, 0x0001},
{0x01D456, 0x0004},
{0x01D49D, 0x0001},
{0x01D49E, 0x0004},
{0x01D4A0, 0x0001},
{0x01D4A2, 0x0004},
{0x01D4A3, 0x0001},
{0x01D4A5, 0x0004},
{0x01D4A7, 0x0001},
{0x01D4A9, 0x0004},
{0x01D4AD, 0x0001},
{0x01D4AE, 0x0004},
{0x01D4BA, 0x0001},
{0x01D4BB, 0x0004},
{0x01D4BC, 0x0001},
{0x01D4BD, 0x0004},
{0x01D4C4, 0x0001},
{0x01D4C5, 0x0004},
{0x01D506, 0x0001},
{0x01D507, 0x0004},
{0x01D50B, 0x0001},
{0x01D50D, 0x0004},
{0x01D515, 0x0001},
{0x01D516, 0x0004},
{0x01D51D, 0x0001},
{0x01D51E, 0x0004},
{0x01D53A, 0x0001},
{0x01D53B, 0x0004},
{0x01D53F, 0x0001},
{0x01D540, 0x0004},
{0x01D545, 0x0001},
{0x01D546, 0x0004},
{0x01D547, 0x0001},
{0x01D54A, 0x0004},
{0x01D551, 0x0001},
{0x01D552, 0x0004},
{0x01D6A6, 0x0001},
{0x01D6A8, 0x0004},
{0x01D6C1, 0x0040},
{0x01D6C2, 0x0004},
{0x01D6DB, 0x0040},
{0x01D6DC, 0x0004},
{0x01D6FB, 0x0040},
{0x01D6FC, 0x0004},
{0x01D715, 0x0040},
{0x01D716, 0x0004},
{0x01D735, 0x0040},
{0x01D736, 0x0004},
{0x01D74F, 0x0040},
{0x01D750, 0x0004},
{0x01D76F, 0x0040},
{0x01D770, 0x0004},
{0x01D789, 0x0040},
{0x01D78A, 0x0004},
{0x01D7A9, 0x0040},
{0x01D7AA, 0x0004},
{0x01D7C3, 0x0040},
{0x01D7C4, 0x0004},
{0x01D7CC, 0x0001},
{0x01D7CE, 0x0002},
{0x01D800, 0x0040},
{0x01DA00, 0x0010},
{0x01DA37, 0x0040},
{0x01DA3B, 0x0010},
{0x01DA6D, 0x0040},
{0x01DA75, 0x0010},
{0x01DA76, 0x0040},
{0x01DA84, 0x0010},
{0x01DA85, 0x0040},
{0x01DA87, 0x0020},
{0x01DA8C, 0x0001},
{0x01DA9B, 0x0010},
{0x01DAA0, 0x0001},
{0x01DAA1, 0x0010},
{0x01DAB0, 0x0001},
{0x01DF00, 0x0004},
{0x01DF1F, 0x0001},
{0x01DF25, 0x0004},
{0x01DF2B, 0x0001},
{0x01E000, 0x0010},
{0x01E007, 0x0001},
{0x01E008, 0x0010},
{0x01E019, 0x0001},
{0x01E01B, 0x0010},
{0x01E022, 0x0001},
{0x01E023, 0x0010},
{0x01E025, 0x0001},
{0x01E026, 0x0010},
{0x01E02B, 0x0001},
{0x01E030, 0x0004},
{0x01E06E, 0x0001},
{0x01E08F, 0x0010},
{0x01E090, 0x0001},
{0x01E100, 0x0004},
{0x01E12D, 0x0001},
{0x01E130, 0x0010},
{0x01E137, 0x0004},
{0x01E13E, 0x0001},
{0x01E140, 0x0002},
{0x01E14A, 0x0001},
{0x01E14E, 0x0004},
{0x01E14F, 0x0040},
{0x01E150, 0x0001},
{0x01E290, 0x0004},
{0x01E2AE, 0x0010},
{0x01E2AF, 0x0001},
{0x01E2C0, 0x0004},
{0x01E2EC, 0x0010},
{0x01E2F0, 0x0002},
{0x01E2FA, 0x0001},
{0x01E2FF, 0x0040},
{0x01E300, 0x0001},
{0x01E4D0, 0x0004},
{0x01E4EC, 0x0010},
{0x01E4F0, 0x0002},
{0x01E4FA, 0x0001},
{0x01E7E0, 0x0004},
{0x01E7E7, 0x0001},
{0x01E7E8, 0x0004},
{0x01E7EC, 0x0001},
{0x01E7ED, 0x0004},
{0x01E7EF, 0x0001},
{0x01E7F0, 0x0004},
{0x01E7FF, 0x0001},
{0x01E800, 0x0004},
{0x01E8C5, 0x0001},
{0x01E8C7, 0x0002},
{0x01E8D0, 0x0010},
{0x01E8D7, 0x0001},
{0x01E900, 0x0004},
{0x01E944, 0x0010},
{0x01E94B, 0x0004},
{0x01E94C, 0x0001},
{0x01E950, 0x0002},
{0x01E95A, 0x0001},
{0x01E95E, 0x0020},
{0x01E960, 0x0001},
{0x01EC71, 0x0002},
{0x01ECAC, 0x0040},
{0x01ECAD, 0x0002},
{0x01ECB0, 0x0040},
{0x01ECB1, 0x0002},
{0x01ECB5, 0x0001},
{0x01ED01, 0x0002},
{0x01ED2E, 0x0040},
{0x01ED2F, 0x0002},
{0x01ED3E, 0x0001},
{0x01EE00, 0x0004},
{0x01EE04, 0x0001},
{0x01EE05, 0x0004},
{0x01EE20, 0x0001},
{0x01EE21, 0x0004},
{0x01EE23, 0x0001},
{0x01EE24, 0x0004},
{0x01EE25, 0x0001},
{0x01EE27, 0x0004},
{0x01EE28, 0x0001},
{0x01EE29, 0x0004},
{0x01EE33, 0x0001},
{0x01EE34, 0x0004},
{0x01EE38, 0x0001},
{0x01EE39, 0x0004},
{0x01EE3A, 0x0001},
{0x01EE3B, 0x0004},
{0x01EE3C, 0x0001},
{0x01EE42, 0x0004},
{0x01EE43, 0x0001},
{0x01EE47, 0x0004},
{0x01EE48, 0x0001},
{0x01EE49, 0x0004},
{0x01EE4A, 0x0001},
{0x01EE4B, 0x0004},
{0x01EE4C, 0x0001},
{0x01EE4D, 0x0004},
{0x01EE50, 0x0001},
{0x01EE51, 0x0004},
{0x01EE53, 0x0001},
{0x01EE54, 0x0004},
{0x01EE55, 0x0001},
{0x01EE57, 0x0004},
{0x01EE58, 0x0001},
{0x01EE59, 0x0004},
{0x01EE5A, 0x0001},
{0x01EE5B, 0x0004},
{0x01EE5C, 0x0001},
{0x01EE5D, 0x0004},
{0x01EE5E, 0x0001},
{0x01EE5F, 0x0004},
{0x01EE60, 0x0001},
{0x01EE61, 0x0004},
{0x01EE63, 0x0001},
{0x01EE64, 0x0004},
{0x01EE65, 0x0001},
{0x01EE67, 0x0004},
{0x01EE6B, 0x0001},
{0x01EE6C, 0x0004},
{0x01EE73, 0x0001},
{0x01EE74, 0x0004},
{0x01EE78, 0x0001},
{0x01EE79, 0x0004},
{0x01EE7D, 0x0001},
{0x01EE7E, 0x0004},
{0x01EE7F, 0x0001},
{0x01EE80, 0x0004},
{0x01EE8A, 0x0001},
{0x01EE8B, 0x0004},
{0x01EE9C, 0x0001},
{0x01EEA1, 0x0004},
{0x01EEA4, 0x0001},
{0x01EEA5, 0x0004},
{0x01EEAA, 0x0001},
{0x01EEAB, 0x0004},
{0x01EEBC, 0x0001},
{0x01EEF0, 0x0040},
{0x01EEF2, 0x0001},
{0x01F000, 0x0040},
{0x01F02C, 0x0001},
{0x01F030, 0x0040},
{0x01F094, 0x0001},
{0x01F0A0, 0x0040},
{0x01F0AF, 0x0001},
{0x01F0B1, 0x0040},
{0x01F0C0, 0x0001},
{0x01F0C1, 0x0040},
{0x01F0D0, 0x0001},
{0x01F0D1, 0x0040},
{0x01F0F6, 0x0001},
{0x01F100, 0x0002},
{0x01F10D, 0x0040},
{0x01F1AE, 0x0001},
{0x01F1E6, 0x0040},
{0x01F203, 0x0001},
{0x01F210, 0x0040},
{0x01F23C, 0x0001},
{0x01F240, 0x0040},
{0x01F249, 0x0001},
{0x01F250, 0x0040},
{0x01F252, 0x0001},
{0x01F260, 0x0040},
{0x01F266, 0x0001},
{0x01F300, 0x0040},
{0x01F6D8, 0x0001},
{0x01F6DC, 0x0040},
{0x01F6ED, 0x0001},
{0x01F6F0, 0x0040},
{0x01F6FD, 0x0001},
{0x01F700, 0x0040},
{0x01F777, 0x0001},
{0x01F77B, 0x0040},
{0x01F7DA, 0x0001},
{0x01F7E0, 0x0040},
{0x01F7EC, 0x0001},
{0x01F7F0, 0x0040},
{0x01F7F1, 0x0001},
{0x01F800, 0x0040},
{0x01F80C, 0x0001},
{0x01F810, 0x0040},
{0x01F848, 0x0001},
{0x01F850, 0x0040},
{0x01F85A, 0x0001},
{0x01F860, 0x0040},
{0x01F888, 0x0001},
{0x01F890, 0x0040},
{0x01F8AE, 0x0001},
{0x01F8B0, 0x0040},
{0x01F8B2, 0x0001},
{0x01F900, 0x0040},
{0x01FA54, 0x0001},
{0x01FA60, 0x0040},
{0x01FA6E, 0x0001},
{0x01FA70, 0x0040},
{0x01FA7D, 0x0001},
{0x01FA80, 0x0040},
{0x01FA89, 0x0001},
{0x01FA90, 0x0040},
{0x01FABE, 0x0001},
{0x01FABF, 0x0040},
{0x01FAC6, 0x0001},
{0x01FACE, 0x0040},
{0x01FADC, 0x0001},
{0x01FAE0, 0x0040},
{0x01FAE9, 0x0001},
{0x01FAF0, 0x0040},
{0x01FAF9, 0x0001},
{0x01FB00, 0x0040},
{0x01FB93, 0x0001},
{0x01FB94, 0x0040},
{0x01FBCB, 0x0001},
{0x01FBF0, 0x0002},
{0x01FBFA, 0x0001},
{0x020000, 0x0004},
{0x02A6E0, 0x0001},
{0x02A700, 0x0004},
{0x02B73A, 0x0001},
{0x02B740, 0x0004},
{0x02B81E, 0x0001},
{0x02B820, 0x0004},
{0x02CEA2, 0x0001},
{0x02CEB0, 0x0004},
{0x02EBE1, 0x0001},
{0x02EBF0, 0x0004},
{0x02EE5E, 0x0001},
{0x02F800, 0x0004},
{0x02FA1E, 0x0001},
{0x030000, 0x0004},
{0x03134B, 0x0001},
{0x031350, 0x0004},
{0x0323B0, 0x0001},
{0x0E0001, 0x0080},
{0x0E0002, 0x0001},
{0x0E0020, 0x0080},
{0x0E0080, 0x0001},
{0x0E0100, 0x0010},
{0x0E01F0, 0x0001},
{0x0F0000, 0x0080},
{0x0FFFFE, 0x0001},
{0x100000, 0x0080},
{0x10FFFE, 0x0001},
{0x110000, 0x0000},
};

const std::unordered_set<uint32_t> unicode_set_whitespace = {
0x000009,
0x00000A,
0x00000B,
0x00000C,
0x00000D,
0x000020,
0x000085,
0x0000A0,
0x001680,
0x002000,
0x002001,
0x002002,
0x002003,
0x002004,
0x002005,
0x002006,
0x002007,
0x002008,
0x002009,
0x00200A,
0x002028,
0x002029,
0x00202F,
0x00205F,
0x003000,
};

// list is always in ascending order, to enable binary search
const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase = {
{0x000041, 0x000061},
{0x000042, 0x000062},
{0x000043, 0x000063},
{0x000044, 0x000064},
{0x000045, 0x000065},
{0x000046, 0x000066},
{0x000047, 0x000067},
{0x000048, 0x000068},
{0x000049, 0x000069},
{0x00004A, 0x00006A},
{0x00004B, 0x00006B},
{0x00004C, 0x00006C},
{0x00004D, 0x00006D},
{0x00004E, 0x00006E},
{0x00004F, 0x00006F},
{0x000050, 0x000070},
{0x000051, 0x000071},
{0x000052, 0x000072},
{0x000053, 0x000073},
{0x000054, 0x000074},
{0x000055, 0x000075},
{0x000056, 0x000076},
{0x000057, 0x000077},
{0x000058, 0x000078},
{0x000059, 0x000079},
{0x00005A, 0x00007A},
{0x0000C0, 0x0000E0},
{0x0000C1, 0x0000E1},
{0x0000C2, 0x0000E2},
{0x0000C3, 0x0000E3},
{0x0000C4, 0x0000E4},
{0x0000C5, 0x0000E5},
{0x0000C6, 0x0000E6},
{0x0000C7, 0x0000E7},
{0x0000C8, 0x0000E8},
{0x0000C9, 0x0000E9},
{0x0000CA, 0x0000EA},
{0x0000CB, 0x0000EB},
{0x0000CC, 0x0000EC},
{0x0000CD, 0x0000ED},
{0x0000CE, 0x0000EE},
{0x0000CF, 0x0000EF},
{0x0000D0, 0x0000F0},
{0x0000D1, 0x0000F1},
{0x0000D2, 0x0000F2},
{0x0000D3, 0x0000F3},
{0x0000D4, 0x0000F4},
{0x0000D5, 0x0000F5},
{0x0000D6, 0x0000F6},
{0x0000D8, 0x0000F8},
{0x0000D9, 0x0000F9},
{0x0000DA, 0x0000FA},
{0x0000DB, 0x0000FB},
{0x0000DC, 0x0000FC},
{0x0000DD, 0x0000FD},
{0x0000DE, 0x0000FE},
{0x000100, 0x000101},
{0x000102, 0x000103},
{0x000104, 0x000105},
{0x000106, 0x000107},
{0x000108, 0x000109},
{0x00010A, 0x00010B},
{0x00010C, 0x00010D},
{0x00010E, 0x00010F},
{0x000110, 0x000111},
{0x000112, 0x000113},
{0x000114, 0x000115},
{0x000116, 0x000117},
{0x000118, 0x000119},
{0x00011A, 0x00011B},
{0x00011C, 0x00011D},
{0x00011E, 0x00011F},
{0x000120, 0x000121},
{0x000122, 0x000123},
{0x000124, 0x000125},
{0x000126, 0x000127},
{0x000128, 0x000129},
{0x00012A, 0x00012B},
{0x00012C, 0x00012D},
{0x00012E, 0x00012F},
{0x000130, 0x000069},
{0x000132, 0x000133},
{0x000134, 0x000135},
{0x000136, 0x000137},
{0x000139, 0x00013A},
{0x00013B, 0x00013C},
{0x00013D, 0x00013E},
{0x00013F, 0x000140},
{0x000141, 0x000142},
{0x000143, 0x000144},
{0x000145, 0x000146},
{0x000147, 0x000148},
{0x00014A, 0x00014B},
{0x00014C, 0x00014D},
{0x00014E, 0x00014F},
{0x000150, 0x000151},
{0x000152, 0x000153},
{0x000154, 0x000155},
{0x000156, 0x000157},
{0x000158, 0x000159},
{0x00015A, 0x00015B},
{0x00015C, 0x00015D},
{0x00015E, 0x00015F},
{0x000160, 0x000161},
{0x000162, 0x000163},
{0x000164, 0x000165},
{0x000166, 0x000167},
{0x000168, 0x000169},
{0x00016A, 0x00016B},
{0x00016C, 0x00016D},
{0x00016E, 0x00016F},
{0x000170, 0x000171},
{0x000172, 0x000173},
{0x000174, 0x000175},
{0x000176, 0x000177},
{0x000178, 0x0000FF},
{0x000179, 0x00017A},
{0x00017B, 0x00017C},
{0x00017D, 0x00017E},
{0x000181, 0x000253},
{0x000182, 0x000183},
{0x000184, 0x000185},
{0x000186, 0x000254},
{0x000187, 0x000188},
{0x000189, 0x000256},
{0x00018A, 0x000257},
{0x00018B, 0x00018C},
{0x00018E, 0x0001DD},
{0x00018F, 0x000259},
{0x000190, 0x00025B},
{0x000191, 0x000192},
{0x000193, 0x000260},
{0x000194, 0x000263},
{0x000196, 0x000269},
{0x000197, 0x000268},
{0x000198, 0x000199},
{0x00019C, 0x00026F},
{0x00019D, 0x000272},
{0x00019F, 0x000275},
{0x0001A0, 0x0001A1},
{0x0001A2, 0x0001A3},
{0x0001A4, 0x0001A5},
{0x0001A6, 0x000280},
{0x0001A7, 0x0001A8},
{0x0001A9, 0x000283},
{0x0001AC, 0x0001AD},
{0x0001AE, 0x000288},
{0x0001AF, 0x0001B0},
{0x0001B1, 0x00028A},
{0x0001B2, 0x00028B},
{0x0001B3, 0x0001B4},
{0x0001B5, 0x0001B6},
{0x0001B7, 0x000292},
{0x0001B8, 0x0001B9},
{0x0001BC, 0x0001BD},
{0x0001C4, 0x0001C6},
{0x0001C5, 0x0001C6},
{0x0001C7, 0x0001C9},
{0x0001C8, 0x0001C9},
{0x0001CA, 0x0001CC},
{0x0001CB, 0x0001CC},
{0x0001CD, 0x0001CE},
{0x0001CF, 0x0001D0},
{0x0001D1, 0x0001D2},
{0x0001D3, 0x0001D4},
{0x0001D5, 0x0001D6},
{0x0001D7, 0x0001D8},
{0x0001D9, 0x0001DA},
{0x0001DB, 0x0001DC},
{0x0001DE, 0x0001DF},
{0x0001E0, 0x0001E1},
{0x0001E2, 0x0001E3},
{0x0001E4, 0x0001E5},
{0x0001E6, 0x0001E7},
{0x0001E8, 0x0001E9},
{0x0001EA, 0x0001EB},
{0x0001EC, 0x0001ED},
{0x0001EE, 0x0001EF},
{0x0001F1, 0x0001F3},
{0x0001F2, 0x0001F3},
{0x0001F4, 0x0001F5},
{0x0001F6, 0x000195},
{0x0001F7, 0x0001BF},
{0x0001F8, 0x0001F9},
{0x0001FA, 0x0001FB},
{0x0001FC, 0x0001FD},
{0x0001FE, 0x0001FF},
{0x000200, 0x000201},
{0x000202, 0x000203},
{0x000204, 0x000205},
{0x000206, 0x000207},
{0x000208, 0x000209},
{0x00020A, 0x00020B},
{0x00020C, 0x00020D},
{0x00020E, 0x00020F},
{0x000210, 0x000211},
{0x000212, 0x000213},
{0x000214, 0x000215},
{0x000216, 0x000217},
{0x000218, 0x000219},
{0x00021A, 0x00021B},
{0x00021C, 0x00021D},
{0x00021E, 0x00021F},
{0x000220, 0x00019E},
{0x000222, 0x000223},
{0x000224, 0x000225},
{0x000226, 0x000227},
{0x000228, 0x000229},
{0x00022A, 0x00022B},
{0x00022C, 0x00022D},
{0x00022E, 0x00022F},
{0x000230, 0x000231},
{0x000232, 0x000233},
{0x00023A, 0x002C65},
{0x00023B, 0x00023C},
{0x00023D, 0x00019A},
{0x00023E, 0x002C66},
{0x000241, 0x000242},
{0x000243, 0x000180},
{0x000244, 0x000289},
{0x000245, 0x00028C},
{0x000246, 0x000247},
{0x000248, 0x000249},
{0x00024A, 0x00024B},
{0x00024C, 0x00024D},
{0x00024E, 0x00024F},
{0x000370, 0x000371},
{0x000372, 0x000373},
{0x000376, 0x000377},
{0x00037F, 0x0003F3},
{0x000386, 0x0003AC},
{0x000388, 0x0003AD},
{0x000389, 0x0003AE},
{0x00038A, 0x0003AF},
{0x00038C, 0x0003CC},
{0x00038E, 0x0003CD},
{0x00038F, 0x0003CE},
{0x000391, 0x0003B1},
{0x000392, 0x0003B2},
{0x000393, 0x0003B3},
{0x000394, 0x0003B4},
{0x000395, 0x0003B5},
{0x000396, 0x0003B6},
{0x000397, 0x0003B7},
{0x000398, 0x0003B8},
{0x000399, 0x0003B9},
{0x00039A, 0x0003BA},
{0x00039B, 0x0003BB},
{0x00039C, 0x0003BC},
{0x00039D, 0x0003BD},
{0x00039E, 0x0003BE},
{0x00039F, 0x0003BF},
{0x0003A0, 0x0003C0},
{0x0003A1, 0x0003C1},
{0x0003A3, 0x0003C3},
{0x0003A4, 0x0003C4},
{0x0003A5, 0x0003C5},
{0x0003A6, 0x0003C6},
{0x0003A7, 0x0003C7},
{0x0003A8, 0x0003C8},
{0x0003A9, 0x0003C9},
{0x0003AA, 0x0003CA},
{0x0003AB, 0x0003CB},
{0x0003CF, 0x0003D7},
{0x0003D8, 0x0003D9},
{0x0003DA, 0x0003DB},
{0x0003DC, 0x0003DD},
{0x0003DE, 0x0003DF},
{0x0003E0, 0x0003E1},
{0x0003E2, 0x0003E3},
{0x0003E4, 0x0003E5},
{0x0003E6, 0x0003E7},
{0x0003E8, 0x0003E9},
{0x0003EA, 0x0003EB},
{0x0003EC, 0x0003ED},
{0x0003EE, 0x0003EF},
{0x0003F4, 0x0003B8},
{0x0003F7, 0x0003F8},
{0x0003F9, 0x0003F2},
{0x0003FA, 0x0003FB},
{0x0003FD, 0x00037B},
{0x0003FE, 0x00037C},
{0x0003FF, 0x00037D},
{0x000400, 0x000450},
{0x000401, 0x000451},
{0x000402, 0x000452},
{0x000403, 0x000453},
{0x000404, 0x000454},
{0x000405, 0x000455},
{0x000406, 0x000456},
{0x000407, 0x000457},
{0x000408, 0x000458},
{0x000409, 0x000459},
{0x00040A, 0x00045A},
{0x00040B, 0x00045B},
{0x00040C, 0x00045C},
{0x00040D, 0x00045D},
{0x00040E, 0x00045E},
{0x00040F, 0x00045F},
{0x000410, 0x000430},
{0x000411, 0x000431},
{0x000412, 0x000432},
{0x000413, 0x000433},
{0x000414, 0x000434},
{0x000415, 0x000435},
{0x000416, 0x000436},
{0x000417, 0x000437},
{0x000418, 0x000438},
{0x000419, 0x000439},
{0x00041A, 0x00043A},
{0x00041B, 0x00043B},
{0x00041C, 0x00043C},
{0x00041D, 0x00043D},
{0x00041E, 0x00043E},
{0x00041F, 0x00043F},
{0x000420, 0x000440},
{0x000421, 0x000441},
{0x000422, 0x000442},
{0x000423, 0x000443},
{0x000424, 0x000444},
{0x000425, 0x000445},
{0x000426, 0x000446},
{0x000427, 0x000447},
{0x000428, 0x000448},
{0x000429, 0x000449},
{0x00042A, 0x00044A},
{0x00042B, 0x00044B},
{0x00042C, 0x00044C},
{0x00042D, 0x00044D},
{0x00042E, 0x00044E},
{0x00042F, 0x00044F},
{0x000460, 0x000461},
{0x000462, 0x000463},
{0x000464, 0x000465},
{0x000466, 0x000467},
{0x000468, 0x000469},
{0x00046A, 0x00046B},
{0x00046C, 0x00046D},
{0x00046E, 0x00046F},
{0x000470, 0x000471},
{0x000472, 0x000473},
{0x000474, 0x000475},
{0x000476, 0x000477},
{0x000478, 0x000479},
{0x00047A, 0x00047B},
{0x00047C, 0x00047D},
{0x00047E, 0x00047F},
{0x000480, 0x000481},
{0x00048A, 0x00048B},
{0x00048C, 0x00048D},
{0x00048E, 0x00048F},
{0x000490, 0x000491},
{0x000492, 0x000493},
{0x000494, 0x000495},
{0x000496, 0x000497},
{0x000498, 0x000499},
{0x00049A, 0x00049B},
{0x00049C, 0x00049D},
{0x00049E, 0x00049F},
{0x0004A0, 0x0004A1},
{0x0004A2, 0x0004A3},
{0x0004A4, 0x0004A5},
{0x0004A6, 0x0004A7},
{0x0004A8, 0x0004A9},
{0x0004AA, 0x0004AB},
{0x0004AC, 0x0004AD},
{0x0004AE, 0x0004AF},
{0x0004B0, 0x0004B1},
{0x0004B2, 0x0004B3},
{0x0004B4, 0x0004B5},
{0x0004B6, 0x0004B7},
{0x0004B8, 0x0004B9},
{0x0004BA, 0x0004BB},
{0x0004BC, 0x0004BD},
{0x0004BE, 0x0004BF},
{0x0004C0, 0x0004CF},
{0x0004C1, 0x0004C2},
{0x0004C3, 0x0004C4},
{0x0004C5, 0x0004C6},
{0x0004C7, 0x0004C8},
{0x0004C9, 0x0004CA},
{0x0004CB, 0x0004CC},
{0x0004CD, 0x0004CE},
{0x0004D0, 0x0004D1},
{0x0004D2, 0x0004D3},
{0x0004D4, 0x0004D5},
{0x0004D6, 0x0004D7},
{0x0004D8, 0x0004D9},
{0x0004DA, 0x0004DB},
{0x0004DC, 0x0004DD},
{0x0004DE, 0x0004DF},
{0x0004E0, 0x0004E1},
{0x0004E2, 0x0004E3},
{0x0004E4, 0x0004E5},
{0x0004E6, 0x0004E7},
{0x0004E8, 0x0004E9},
{0x0004EA, 0x0004EB},
{0x0004EC, 0x0004ED},
{0x0004EE, 0x0004EF},
{0x0004F0, 0x0004F1},
{0x0004F2, 0x0004F3},
{0x0004F4, 0x0004F5},
{0x0004F6, 0x0004F7},
{0x0004F8, 0x0004F9},
{0x0004FA, 0x0004FB},
{0x0004FC, 0x0004FD},
{0x0004FE, 0x0004FF},
{0x000500, 0x000501},
{0x000502, 0x000503},
{0x000504, 0x000505},
{0x000506, 0x000507},
{0x000508, 0x000509},
{0x00050A, 0x00050B},
{0x00050C, 0x00050D},
{0x00050E, 0x00050F},
{0x000510, 0x000511},
{0x000512, 0x000513},
{0x000514, 0x000515},
{0x000516, 0x000517},
{0x000518, 0x000519},
{0x00051A, 0x00051B},
{0x00051C, 0x00051D},
{0x00051E, 0x00051F},
{0x000520, 0x000521},
{0x000522, 0x000523},
{0x000524, 0x000525},
{0x000526, 0x000527},
{0x000528, 0x000529},
{0x00052A, 0x00052B},
{0x00052C, 0x00052D},
{0x00052E, 0x00052F},
{0x000531, 0x000561},
{0x000532, 0x000562},
{0x000533, 0x000563},
{0x000534, 0x000564},
{0x000535, 0x000565},
{0x000536, 0x000566},
{0x000537, 0x000567},
{0x000538, 0x000568},
{0x000539, 0x000569},
{0x00053A, 0x00056A},
{0x00053B, 0x00056B},
{0x00053C, 0x00056C},
{0x00053D, 0x00056D},
{0x00053E, 0x00056E},
{0x00053F, 0x00056F},
{0x000540, 0x000570},
{0x000541, 0x000571},
{0x000542, 0x000572},
{0x000543, 0x000573},
{0x000544, 0x000574},
{0x000545, 0x000575},
{0x000546, 0x000576},
{0x000547, 0x000577},
{0x000548, 0x000578},
{0x000549, 0x000579},
{0x00054A, 0x00057A},
{0x00054B, 0x00057B},
{0x00054C, 0x00057C},
{0x00054D, 0x00057D},
{0x00054E, 0x00057E},
{0x00054F, 0x00057F},
{0x000550, 0x000580},
{0x000551, 0x000581},
{0x000552, 0x000582},
{0x000553, 0x000583},
{0x000554, 0x000584},
{0x000555, 0x000585},
{0x000556, 0x000586},
{0x0010A0, 0x002D00},
{0x0010A1, 0x002D01},
{0x0010A2, 0x002D02},
{0x0010A3, 0x002D03},
{0x0010A4, 0x002D04},
{0x0010A5, 0x002D05},
{0x0010A6, 0x002D06},
{0x0010A7, 0x002D07},
{0x0010A8, 0x002D08},
{0x0010A9, 0x002D09},
{0x0010AA, 0x002D0A},
{0x0010AB, 0x002D0B},
{0x0010AC, 0x002D0C},
{0x0010AD, 0x002D0D},
{0x0010AE, 0x002D0E},
{0x0010AF, 0x002D0F},
{0x0010B0, 0x002D10},
{0x0010B1, 0x002D11},
{0x0010B2, 0x002D12},
{0x0010B3, 0x002D13},
{0x0010B4, 0x002D14},
{0x0010B5, 0x002D15},
{0x0010B6, 0x002D16},
{0x0010B7, 0x002D17},
{0x0010B8, 0x002D18},
{0x0010B9, 0x002D19},
{0x0010BA, 0x002D1A},
{0x0010BB, 0x002D1B},
{0x0010BC, 0x002D1C},
{0x0010BD, 0x002D1D},
{0x0010BE, 0x002D1E},
{0x0010BF, 0x002D1F},
{0x0010C0, 0x002D20},
{0x0010C1, 0x002D21},
{0x0010C2, 0x002D22},
{0x0010C3, 0x002D23},
{0x0010C4, 0x002D24},
{0x0010C5, 0x002D25},
{0x0010C7, 0x002D27},
{0x0010CD, 0x002D2D},
{0x0013A0, 0x00AB70},
{0x0013A1, 0x00AB71},
{0x0013A2, 0x00AB72},
{0x0013A3, 0x00AB73},
{0x0013A4, 0x00AB74},
{0x0013A5, 0x00AB75},
{0x0013A6, 0x00AB76},
{0x0013A7, 0x00AB77},
{0x0013A8, 0x00AB78},
{0x0013A9, 0x00AB79},
{0x0013AA, 0x00AB7A},
{0x0013AB, 0x00AB7B},
{0x0013AC, 0x00AB7C},
{0x0013AD, 0x00AB7D},
{0x0013AE, 0x00AB7E},
{0x0013AF, 0x00AB7F},
{0x0013B0, 0x00AB80},
{0x0013B1, 0x00AB81},
{0x0013B2, 0x00AB82},
{0x0013B3, 0x00AB83},
{0x0013B4, 0x00AB84},
{0x0013B5, 0x00AB85},
{0x0013B6, 0x00AB86},
{0x0013B7, 0x00AB87},
{0x0013B8, 0x00AB88},
{0x0013B9, 0x00AB89},
{0x0013BA, 0x00AB8A},
{0x0013BB, 0x00AB8B},
{0x0013BC, 0x00AB8C},
{0x0013BD, 0x00AB8D},
{0x0013BE, 0x00AB8E},
{0x0013BF, 0x00AB8F},
{0x0013C0, 0x00AB90},
{0x0013C1, 0x00AB91},
{0x0013C2, 0x00AB92},
{0x0013C3, 0x00AB93},
{0x0013C4, 0x00AB94},
{0x0013C5, 0x00AB95},
{0x0013C6, 0x00AB96},
{0x0013C7, 0x00AB97},
{0x0013C8, 0x00AB98},
{0x0013C9, 0x00AB99},
{0x0013CA, 0x00AB9A},
{0x0013CB, 0x00AB9B},
{0x0013CC, 0x00AB9C},
{0x0013CD, 0x00AB9D},
{0x0013CE, 0x00AB9E},
{0x0013CF, 0x00AB9F},
{0x0013D0, 0x00ABA0},
{0x0013D1, 0x00ABA1},
{0x0013D2, 0x00ABA2},
{0x0013D3, 0x00ABA3},
{0x0013D4, 0x00ABA4},
{0x0013D5, 0x00ABA5},
{0x0013D6, 0x00ABA6},
{0x0013D7, 0x00ABA7},
{0x0013D8, 0x00ABA8},
{0x0013D9, 0x00ABA9},
{0x0013DA, 0x00ABAA},
{0x0013DB, 0x00ABAB},
{0x0013DC, 0x00ABAC},
{0x0013DD, 0x00ABAD},
{0x0013DE, 0x00ABAE},
{0x0013DF, 0x00ABAF},
{0x0013E0, 0x00ABB0},
{0x0013E1, 0x00ABB1},
{0x0013E2, 0x00ABB2},
{0x0013E3, 0x00ABB3},
{0x0013E4, 0x00ABB4},
{0x0013E5, 0x00ABB5},
{0x0013E6, 0x00ABB6},
{0x0013E7, 0x00ABB7},
{0x0013E8, 0x00ABB8},
{0x0013E9, 0x00ABB9},
{0x0013EA, 0x00ABBA},
{0x0013EB, 0x00ABBB},
{0x0013EC, 0x00ABBC},
{0x0013ED, 0x00ABBD},
{0x0013EE, 0x00ABBE},
{0x0013EF, 0x00ABBF},
{0x0013F0, 0x0013F8},
{0x0013F1, 0x0013F9},
{0x0013F2, 0x0013FA},
{0x0013F3, 0x0013FB},
{0x0013F4, 0x0013FC},
{0x0013F5, 0x0013FD},
{0x001C90, 0x0010D0},
{0x001C91, 0x0010D1},
{0x001C92, 0x0010D2},
{0x001C93, 0x0010D3},
{0x001C94, 0x0010D4},
{0x001C95, 0x0010D5},
{0x001C96, 0x0010D6},
{0x001C97, 0x0010D7},
{0x001C98, 0x0010D8},
{0x001C99, 0x0010D9},
{0x001C9A, 0x0010DA},
{0x001C9B, 0x0010DB},
{0x001C9C, 0x0010DC},
{0x001C9D, 0x0010DD},
{0x001C9E, 0x0010DE},
{0x001C9F, 0x0010DF},
{0x001CA0, 0x0010E0},
{0x001CA1, 0x0010E1},
{0x001CA2, 0x0010E2},
{0x001CA3, 0x0010E3},
{0x001CA4, 0x0010E4},
{0x001CA5, 0x0010E5},
{0x001CA6, 0x0010E6},
{0x001CA7, 0x0010E7},
{0x001CA8, 0x0010E8},
{0x001CA9, 0x0010E9},
{0x001CAA, 0x0010EA},
{0x001CAB, 0x0010EB},
{0x001CAC, 0x0010EC},
{0x001CAD, 0x0010ED},
{0x001CAE, 0x0010EE},
{0x001CAF, 0x0010EF},
{0x001CB0, 0x0010F0},
{0x001CB1, 0x0010F1},
{0x001CB2, 0x0010F2},
{0x001CB3, 0x0010F3},
{0x001CB4, 0x0010F4},
{0x001CB5, 0x0010F5},
{0x001CB6, 0x0010F6},
{0x001CB7, 0x0010F7},
{0x001CB8, 0x0010F8},
{0x001CB9, 0x0010F9},
{0x001CBA, 0x0010FA},
{0x001CBD, 0x0010FD},
{0x001CBE, 0x0010FE},
{0x001CBF, 0x0010FF},
{0x001E00, 0x001E01},
{0x001E02, 0x001E03},
{0x001E04, 0x001E05},
{0x001E06, 0x001E07},
{0x001E08, 0x001E09},
{0x001E0A, 0x001E0B},
{0x001E0C, 0x001E0D},
{0x001E0E, 0x001E0F},
{0x001E10, 0x001E11},
{0x001E12, 0x001E13},
{0x001E14, 0x001E15},
{0x001E16, 0x001E17},
{0x001E18, 0x001E19},
{0x001E1A, 0x001E1B},
{0x001E1C, 0x001E1D},
{0x001E1E, 0x001E1F},
{0x001E20, 0x001E21},
{0x001E22, 0x001E23},
{0x001E24, 0x001E25},
{0x001E26, 0x001E27},
{0x001E28, 0x001E29},
{0x001E2A, 0x001E2B},
{0x001E2C, 0x001E2D},
{0x001E2E, 0x001E2F},
{0x001E30, 0x001E31},
{0x001E32, 0x001E33},
{0x001E34, 0x001E35},
{0x001E36, 0x001E37},
{0x001E38, 0x001E39},
{0x001E3A, 0x001E3B},
{0x001E3C, 0x001E3D},
{0x001E3E, 0x001E3F},
{0x001E40, 0x001E41},
{0x001E42, 0x001E43},
{0x001E44, 0x001E45},
{0x001E46, 0x001E47},
{0x001E48, 0x001E49},
{0x001E4A, 0x001E4B},
{0x001E4C, 0x001E4D},
{0x001E4E, 0x001E4F},
{0x001E50, 0x001E51},
{0x001E52, 0x001E53},
{0x001E54, 0x001E55},
{0x001E56, 0x001E57},
{0x001E58, 0x001E59},
{0x001E5A, 0x001E5B},
{0x001E5C, 0x001E5D},
{0x001E5E, 0x001E5F},
{0x001E60, 0x001E61},
{0x001E62, 0x001E63},
{0x001E64, 0x001E65},
{0x001E66, 0x001E67},
{0x001E68, 0x001E69},
{0x001E6A, 0x001E6B},
{0x001E6C, 0x001E6D},
{0x001E6E, 0x001E6F},
{0x001E70, 0x001E71},
{0x001E72, 0x001E73},
{0x001E74, 0x001E75},
{0x001E76, 0x001E77},
{0x001E78, 0x001E79},
{0x001E7A, 0x001E7B},
{0x001E7C, 0x001E7D},
{0x001E7E, 0x001E7F},
{0x001E80, 0x001E81},
{0x001E82, 0x001E83},
{0x001E84, 0x001E85},
{0x001E86, 0x001E87},
{0x001E88, 0x001E89},
{0x001E8A, 0x001E8B},
{0x001E8C, 0x001E8D},
{0x001E8E, 0x001E8F},
{0x001E90, 0x001E91},
{0x001E92, 0x001E93},
{0x001E94, 0x001E95},
{0x001E9E, 0x0000DF},
{0x001EA0, 0x001EA1},
{0x001EA2, 0x001EA3},
{0x001EA4, 0x001EA5},
{0x001EA6, 0x001EA7},
{0x001EA8, 0x001EA9},
{0x001EAA, 0x001EAB},
{0x001EAC, 0x001EAD},
{0x001EAE, 0x001EAF},
{0x001EB0, 0x001EB1},
{0x001EB2, 0x001EB3},
{0x001EB4, 0x001EB5},
{0x001EB6, 0x001EB7},
{0x001EB8, 0x001EB9},
{0x001EBA, 0x001EBB},
{0x001EBC, 0x001EBD},
{0x001EBE, 0x001EBF},
{0x001EC0, 0x001EC1},
{0x001EC2, 0x001EC3},
{0x001EC4, 0x001EC5},
{0x001EC6, 0x001EC7},
{0x001EC8, 0x001EC9},
{0x001ECA, 0x001ECB},
{0x001ECC, 0x001ECD},
{0x001ECE, 0x001ECF},
{0x001ED0, 0x001ED1},
{0x001ED2, 0x001ED3},
{0x001ED4, 0x001ED5},
{0x001ED6, 0x001ED7},
{0x001ED8, 0x001ED9},
{0x001EDA, 0x001EDB},
{0x001EDC, 0x001EDD},
{0x001EDE, 0x001EDF},
{0x001EE0, 0x001EE1},
{0x001EE2, 0x001EE3},
{0x001EE4, 0x001EE5},
{0x001EE6, 0x001EE7},
{0x001EE8, 0x001EE9},
{0x001EEA, 0x001EEB},
{0x001EEC, 0x001EED},
{0x001EEE, 0x001EEF},
{0x001EF0, 0x001EF1},
{0x001EF2, 0x001EF3},
{0x001EF4, 0x001EF5},
{0x001EF6, 0x001EF7},
{0x001EF8, 0x001EF9},
{0x001EFA, 0x001EFB},
{0x001EFC, 0x001EFD},
{0x001EFE, 0x001EFF},
{0x001F08, 0x001F00},
{0x001F09, 0x001F01},
{0x001F0A, 0x001F02},
{0x001F0B, 0x001F03},
{0x001F0C, 0x001F04},
{0x001F0D, 0x001F05},
{0x001F0E, 0x001F06},
{0x001F0F, 0x001F07},
{0x001F18, 0x001F10},
{0x001F19, 0x001F11},
{0x001F1A, 0x001F12},
{0x001F1B, 0x001F13},
{0x001F1C, 0x001F14},
{0x001F1D, 0x001F15},
{0x001F28, 0x001F20},
{0x001F29, 0x001F21},
{0x001F2A, 0x001F22},
{0x001F2B, 0x001F23},
{0x001F2C, 0x001F24},
{0x001F2D, 0x001F25},
{0x001F2E, 0x001F26},
{0x001F2F, 0x001F27},
{0x001F38, 0x001F30},
{0x001F39, 0x001F31},
{0x001F3A, 0x001F32},
{0x001F3B, 0x001F33},
{0x001F3C, 0x001F34},
{0x001F3D, 0x001F35},
{0x001F3E, 0x001F36},
{0x001F3F, 0x001F37},
{0x001F48, 0x001F40},
{0x001F49, 0x001F41},
{0x001F4A, 0x001F42},
{0x001F4B, 0x001F43},
{0x001F4C, 0x001F44},
{0x001F4D, 0x001F45},
{0x001F59, 0x001F51},
{0x001F5B, 0x001F53},
{0x001F5D, 0x001F55},
{0x001F5F, 0x001F57},
{0x001F68, 0x001F60},
{0x001F69, 0x001F61},
{0x001F6A, 0x001F62},
{0x001F6B, 0x001F63},
{0x001F6C, 0x001F64},
{0x001F6D, 0x001F65},
{0x001F6E, 0x001F66},
{0x001F6F, 0x001F67},
{0x001F88, 0x001F80},
{0x001F89, 0x001F81},
{0x001F8A, 0x001F82},
{0x001F8B, 0x001F83},
{0x001F8C, 0x001F84},
{0x001F8D, 0x001F85},
{0x001F8E, 0x001F86},
{0x001F8F, 0x001F87},
{0x001F98, 0x001F90},
{0x001F99, 0x001F91},
{0x001F9A, 0x001F92},
{0x001F9B, 0x001F93},
{0x001F9C, 0x001F94},
{0x001F9D, 0x001F95},
{0x001F9E, 0x001F96},
{0x001F9F, 0x001F97},
{0x001FA8, 0x001FA0},
{0x001FA9, 0x001FA1},
{0x001FAA, 0x001FA2},
{0x001FAB, 0x001FA3},
{0x001FAC, 0x001FA4},
{0x001FAD, 0x001FA5},
{0x001FAE, 0x001FA6},
{0x001FAF, 0x001FA7},
{0x001FB8, 0x001FB0},
{0x001FB9, 0x001FB1},
{0x001FBA, 0x001F70},
{0x001FBB, 0x001F71},
{0x001FBC, 0x001FB3},
{0x001FC8, 0x001F72},
{0x001FC9, 0x001F73},
{0x001FCA, 0x001F74},
{0x001FCB, 0x001F75},
{0x001FCC, 0x001FC3},
{0x001FD8, 0x001FD0},
{0x001FD9, 0x001FD1},
{0x001FDA, 0x001F76},
{0x001FDB, 0x001F77},
{0x001FE8, 0x001FE0},
{0x001FE9, 0x001FE1},
{0x001FEA, 0x001F7A},
{0x001FEB, 0x001F7B},
{0x001FEC, 0x001FE5},
{0x001FF8, 0x001F78},
{0x001FF9, 0x001F79},
{0x001FFA, 0x001F7C},
{0x001FFB, 0x001F7D},
{0x001FFC, 0x001FF3},
{0x002126, 0x0003C9},
{0x00212A, 0x00006B},
{0x00212B, 0x0000E5},
{0x002132, 0x00214E},
{0x002160, 0x002170},
{0x002161, 0x002171},
{0x002162, 0x002172},
{0x002163, 0x002173},
{0x002164, 0x002174},
{0x002165, 0x002175},
{0x002166, 0x002176},
{0x002167, 0x002177},
{0x002168, 0x002178},
{0x002169, 0x002179},
{0x00216A, 0x00217A},
{0x00216B, 0x00217B},
{0x00216C, 0x00217C},
{0x00216D, 0x00217D},
{0x00216E, 0x00217E},
{0x00216F, 0x00217F},
{0x002183, 0x002184},
{0x0024B6, 0x0024D0},
{0x0024B7, 0x0024D1},
{0x0024B8, 0x0024D2},
{0x0024B9, 0x0024D3},
{0x0024BA, 0x0024D4},
{0x0024BB, 0x0024D5},
{0x0024BC, 0x0024D6},
{0x0024BD, 0x0024D7},
{0x0024BE, 0x0024D8},
{0x0024BF, 0x0024D9},
{0x0024C0, 0x0024DA},
{0x0024C1, 0x0024DB},
{0x0024C2, 0x0024DC},
{0x0024C3, 0x0024DD},
{0x0024C4, 0x0024DE},
{0x0024C5, 0x0024DF},
{0x0024C6, 0x0024E0},
{0x0024C7, 0x0024E1},
{0x0024C8, 0x0024E2},
{0x0024C9, 0x0024E3},
{0x0024CA, 0x0024E4},
{0x0024CB, 0x0024E5},
{0x0024CC, 0x0024E6},
{0x0024CD, 0x0024E7},
{0x0024CE, 0x0024E8},
{0x0024CF, 0x0024E9},
{0x002C00, 0x002C30},
{0x002C01, 0x002C31},
{0x002C02, 0x002C32},
{0x002C03, 0x002C33},
{0x002C04, 0x002C34},
{0x002C05, 0x002C35},
{0x002C06, 0x002C36},
{0x002C07, 0x002C37},
{0x002C08, 0x002C38},
{0x002C09, 0x002C39},
{0x002C0A, 0x002C3A},
{0x002C0B, 0x002C3B},
{0x002C0C, 0x002C3C},
{0x002C0D, 0x002C3D},
{0x002C0E, 0x002C3E},
{0x002C0F, 0x002C3F},
{0x002C10, 0x002C40},
{0x002C11, 0x002C41},
{0x002C12, 0x002C42},
{0x002C13, 0x002C43},
{0x002C14, 0x002C44},
{0x002C15, 0x002C45},
{0x002C16, 0x002C46},
{0x002C17, 0x002C47},
{0x002C18, 0x002C48},
{0x002C19, 0x002C49},
{0x002C1A, 0x002C4A},
{0x002C1B, 0x002C4B},
{0x002C1C, 0x002C4C},
{0x002C1D, 0x002C4D},
{0x002C1E, 0x002C4E},
{0x002C1F, 0x002C4F},
{0x002C20, 0x002C50},
{0x002C21, 0x002C51},
{0x002C22, 0x002C52},
{0x002C23, 0x002C53},
{0x002C24, 0x002C54},
{0x002C25, 0x002C55},
{0x002C26, 0x002C56},
{0x002C27, 0x002C57},
{0x002C28, 0x002C58},
{0x002C29, 0x002C59},
{0x002C2A, 0x002C5A},
{0x002C2B, 0x002C5B},
{0x002C2C, 0x002C5C},
{0x002C2D, 0x002C5D},
{0x002C2E, 0x002C5E},
{0x002C2F, 0x002C5F},
{0x002C60, 0x002C61},
{0x002C62, 0x00026B},
{0x002C63, 0x001D7D},
{0x002C64, 0x00027D},
{0x002C67, 0x002C68},
{0x002C69, 0x002C6A},
{0x002C6B, 0x002C6C},
{0x002C6D, 0x000251},
{0x002C6E, 0x000271},
{0x002C6F, 0x000250},
{0x002C70, 0x000252},
{0x002C72, 0x002C73},
{0x002C75, 0x002C76},
{0x002C7E, 0x00023F},
{0x002C7F, 0x000240},
{0x002C80, 0x002C81},
{0x002C82, 0x002C83},
{0x002C84, 0x002C85},
{0x002C86, 0x002C87},
{0x002C88, 0x002C89},
{0x002C8A, 0x002C8B},
{0x002C8C, 0x002C8D},
{0x002C8E, 0x002C8F},
{0x002C90, 0x002C91},
{0x002C92, 0x002C93},
{0x002C94, 0x002C95},
{0x002C96, 0x002C97},
{0x002C98, 0x002C99},
{0x002C9A, 0x002C9B},
{0x002C9C, 0x002C9D},
{0x002C9E, 0x002C9F},
{0x002CA0, 0x002CA1},
{0x002CA2, 0x002CA3},
{0x002CA4, 0x002CA5},
{0x002CA6, 0x002CA7},
{0x002CA8, 0x002CA9},
{0x002CAA, 0x002CAB},
{0x002CAC, 0x002CAD},
{0x002CAE, 0x002CAF},
{0x002CB0, 0x002CB1},
{0x002CB2, 0x002CB3},
{0x002CB4, 0x002CB5},
{0x002CB6, 0x002CB7},
{0x002CB8, 0x002CB9},
{0x002CBA, 0x002CBB},
{0x002CBC, 0x002CBD},
{0x002CBE, 0x002CBF},
{0x002CC0, 0x002CC1},
{0x002CC2, 0x002CC3},
{0x002CC4, 0x002CC5},
{0x002CC6, 0x002CC7},
{0x002CC8, 0x002CC9},
{0x002CCA, 0x002CCB},
{0x002CCC, 0x002CCD},
{0x002CCE, 0x002CCF},
{0x002CD0, 0x002CD1},
{0x002CD2, 0x002CD3},
{0x002CD4, 0x002CD5},
{0x002CD6, 0x002CD7},
{0x002CD8, 0x002CD9},
{0x002CDA, 0x002CDB},
{0x002CDC, 0x002CDD},
{0x002CDE, 0x002CDF},
{0x002CE0, 0x002CE1},
{0x002CE2, 0x002CE3},
{0x002CEB, 0x002CEC},
{0x002CED, 0x002CEE},
{0x002CF2, 0x002CF3},
{0x00A640, 0x00A641},
{0x00A642, 0x00A643},
{0x00A644, 0x00A645},
{0x00A646, 0x00A647},
{0x00A648, 0x00A649},
{0x00A64A, 0x00A64B},
{0x00A64C, 0x00A64D},
{0x00A64E, 0x00A64F},
{0x00A650, 0x00A651},
{0x00A652, 0x00A653},
{0x00A654, 0x00A655},
{0x00A656, 0x00A657},
{0x00A658, 0x00A659},
{0x00A65A, 0x00A65B},
{0x00A65C, 0x00A65D},
{0x00A65E, 0x00A65F},
{0x00A660, 0x00A661},
{0x00A662, 0x00A663},
{0x00A664, 0x00A665},
{0x00A666, 0x00A667},
{0x00A668, 0x00A669},
{0x00A66A, 0x00A66B},
{0x00A66C, 0x00A66D},
{0x00A680, 0x00A681},
{0x00A682, 0x00A683},
{0x00A684, 0x00A685},
{0x00A686, 0x00A687},
{0x00A688, 0x00A689},
{0x00A68A, 0x00A68B},
{0x00A68C, 0x00A68D},
{0x00A68E, 0x00A68F},
{0x00A690, 0x00A691},
{0x00A692, 0x00A693},
{0x00A694, 0x00A695},
{0x00A696, 0x00A697},
{0x00A698, 0x00A699},
{0x00A69A, 0x00A69B},
{0x00A722, 0x00A723},
{0x00A724, 0x00A725},
{0x00A726, 0x00A727},
{0x00A728, 0x00A729},
{0x00A72A, 0x00A72B},
{0x00A72C, 0x00A72D},
{0x00A72E, 0x00A72F},
{0x00A732, 0x00A733},
{0x00A734, 0x00A735},
{0x00A736, 0x00A737},
{0x00A738, 0x00A739},
{0x00A73A, 0x00A73B},
{0x00A73C, 0x00A73D},
{0x00A73E, 0x00A73F},
{0x00A740, 0x00A741},
{0x00A742, 0x00A743},
{0x00A744, 0x00A745},
{0x00A746, 0x00A747},
{0x00A748, 0x00A749},
{0x00A74A, 0x00A74B},
{0x00A74C, 0x00A74D},
{0x00A74E, 0x00A74F},
{0x00A750, 0x00A751},
{0x00A752, 0x00A753},
{0x00A754, 0x00A755},
{0x00A756, 0x00A757},
{0x00A758, 0x00A759},
{0x00A75A, 0x00A75B},
{0x00A75C, 0x00A75D},
{0x00A75E, 0x00A75F},
{0x00A760, 0x00A761},
{0x00A762, 0x00A763},
{0x00A764, 0x00A765},
{0x00A766, 0x00A767},
{0x00A768, 0x00A769},
{0x00A76A, 0x00A76B},
{0x00A76C, 0x00A76D},
{0x00A76E, 0x00A76F},
{0x00A779, 0x00A77A},
{0x00A77B, 0x00A77C},
{0x00A77D, 0x001D79},
{0x00A77E, 0x00A77F},
{0x00A780, 0x00A781},
{0x00A782, 0x00A783},
{0x00A784, 0x00A785},
{0x00A786, 0x00A787},
{0x00A78B, 0x00A78C},
{0x00A78D, 0x000265},
{0x00A790, 0x00A791},
{0x00A792, 0x00A793},
{0x00A796, 0x00A797},
{0x00A798, 0x00A799},
{0x00A79A, 0x00A79B},
{0x00A79C, 0x00A79D},
{0x00A79E, 0x00A79F},
{0x00A7A0, 0x00A7A1},
{0x00A7A2, 0x00A7A3},
{0x00A7A4, 0x00A7A5},
{0x00A7A6, 0x00A7A7},
{0x00A7A8, 0x00A7A9},
{0x00A7AA, 0x000266},
{0x00A7AB, 0x00025C},
{0x00A7AC, 0x000261},
{0x00A7AD, 0x00026C},
{0x00A7AE, 0x00026A},
{0x00A7B0, 0x00029E},
{0x00A7B1, 0x000287},
{0x00A7B2, 0x00029D},
{0x00A7B3, 0x00AB53},
{0x00A7B4, 0x00A7B5},
{0x00A7B6, 0x00A7B7},
{0x00A7B8, 0x00A7B9},
{0x00A7BA, 0x00A7BB},
{0x00A7BC, 0x00A7BD},
{0x00A7BE, 0x00A7BF},
{0x00A7C0, 0x00A7C1},
{0x00A7C2, 0x00A7C3},
{0x00A7C4, 0x00A794},
{0x00A7C5, 0x000282},
{0x00A7C6, 0x001D8E},
{0x00A7C7, 0x00A7C8},
{0x00A7C9, 0x00A7CA},
{0x00A7D0, 0x00A7D1},
{0x00A7D6, 0x00A7D7},
{0x00A7D8, 0x00A7D9},
{0x00A7F5, 0x00A7F6},
{0x00FF21, 0x00FF41},
{0x00FF22, 0x00FF42},
{0x00FF23, 0x00FF43},
{0x00FF24, 0x00FF44},
{0x00FF25, 0x00FF45},
{0x00FF26, 0x00FF46},
{0x00FF27, 0x00FF47},
{0x00FF28, 0x00FF48},
{0x00FF29, 0x00FF49},
{0x00FF2A, 0x00FF4A},
{0x00FF2B, 0x00FF4B},
{0x00FF2C, 0x00FF4C},
{0x00FF2D, 0x00FF4D},
{0x00FF2E, 0x00FF4E},
{0x00FF2F, 0x00FF4F},
{0x00FF30, 0x00FF50},
{0x00FF31, 0x00FF51},
{0x00FF32, 0x00FF52},
{0x00FF33, 0x00FF53},
{0x00FF34, 0x00FF54},
{0x00FF35, 0x00FF55},
{0x00FF36, 0x00FF56},
{0x00FF37, 0x00FF57},
{0x00FF38, 0x00FF58},
{0x00FF39, 0x00FF59},
{0x00FF3A, 0x00FF5A},
{0x010400, 0x010428},
{0x010401, 0x010429},
{0x010402, 0x01042A},
{0x010403, 0x01042B},
{0x010404, 0x01042C},
{0x010405, 0x01042D},
{0x010406, 0x01042E},
{0x010407, 0x01042F},
{0x010408, 0x010430},
{0x010409, 0x010431},
{0x01040A, 0x010432},
{0x01040B, 0x010433},
{0x01040C, 0x010434},
{0x01040D, 0x010435},
{0x01040E, 0x010436},
{0x01040F, 0x010437},
{0x010410, 0x010438},
{0x010411, 0x010439},
{0x010412, 0x01043A},
{0x010413, 0x01043B},
{0x010414, 0x01043C},
{0x010415, 0x01043D},
{0x010416, 0x01043E},
{0x010417, 0x01043F},
{0x010418, 0x010440},
{0x010419, 0x010441},
{0x01041A, 0x010442},
{0x01041B, 0x010443},
{0x01041C, 0x010444},
{0x01041D, 0x010445},
{0x01041E, 0x010446},
{0x01041F, 0x010447},
{0x010420, 0x010448},
{0x010421, 0x010449},
{0x010422, 0x01044A},
{0x010423, 0x01044B},
{0x010424, 0x01044C},
{0x010425, 0x01044D},
{0x010426, 0x01044E},
{0x010427, 0x01044F},
{0x0104B0, 0x0104D8},
{0x0104B1, 0x0104D9},
{0x0104B2, 0x0104DA},
{0x0104B3, 0x0104DB},
{0x0104B4, 0x0104DC},
{0x0104B5, 0x0104DD},
{0x0104B6, 0x0104DE},
{0x0104B7, 0x0104DF},
{0x0104B8, 0x0104E0},
{0x0104B9, 0x0104E1},
{0x0104BA, 0x0104E2},
{0x0104BB, 0x0104E3},
{0x0104BC, 0x0104E4},
{0x0104BD, 0x0104E5},
{0x0104BE, 0x0104E6},
{0x0104BF, 0x0104E7},
{0x0104C0, 0x0104E8},
{0x0104C1, 0x0104E9},
{0x0104C2, 0x0104EA},
{0x0104C3, 0x0104EB},
{0x0104C4, 0x0104EC},
{0x0104C5, 0x0104ED},
{0x0104C6, 0x0104EE},
{0x0104C7, 0x0104EF},
{0x0104C8, 0x0104F0},
{0x0104C9, 0x0104F1},
{0x0104CA, 0x0104F2},
{0x0104CB, 0x0104F3},
{0x0104CC, 0x0104F4},
{0x0104CD, 0x0104F5},
{0x0104CE, 0x0104F6},
{0x0104CF, 0x0104F7},
{0x0104D0, 0x0104F8},
{0x0104D1, 0x0104F9},
{0x0104D2, 0x0104FA},
{0x0104D3, 0x0104FB},
{0x010570, 0x010597},
{0x010571, 0x010598},
{0x010572, 0x010599},
{0x010573, 0x01059A},
{0x010574, 0x01059B},
{0x010575, 0x01059C},
{0x010576, 0x01059D},
{0x010577, 0x01059E},
{0x010578, 0x01059F},
{0x010579, 0x0105A0},
{0x01057A, 0x0105A1},
{0x01057C, 0x0105A3},
{0x01057D, 0x0105A4},
{0x01057E, 0x0105A5},
{0x01057F, 0x0105A6},
{0x010580, 0x0105A7},
{0x010581, 0x0105A8},
{0x010582, 0x0105A9},
{0x010583, 0x0105AA},
{0x010584, 0x0105AB},
{0x010585, 0x0105AC},
{0x010586, 0x0105AD},
{0x010587, 0x0105AE},
{0x010588, 0x0105AF},
{0x010589, 0x0105B0},
{0x01058A, 0x0105B1},
{0x01058C, 0x0105B3},
{0x01058D, 0x0105B4},
{0x01058E, 0x0105B5},
{0x01058F, 0x0105B6},
{0x010590, 0x0105B7},
{0x010591, 0x0105B8},
{0x010592, 0x0105B9},
{0x010594, 0x0105BB},
{0x010595, 0x0105BC},
{0x010C80, 0x010CC0},
{0x010C81, 0x010CC1},
{0x010C82, 0x010CC2},
{0x010C83, 0x010CC3},
{0x010C84, 0x010CC4},
{0x010C85, 0x010CC5},
{0x010C86, 0x010CC6},
{0x010C87, 0x010CC7},
{0x010C88, 0x010CC8},
{0x010C89, 0x010CC9},
{0x010C8A, 0x010CCA},
{0x010C8B, 0x010CCB},
{0x010C8C, 0x010CCC},
{0x010C8D, 0x010CCD},
{0x010C8E, 0x010CCE},
{0x010C8F, 0x010CCF},
{0x010C90, 0x010CD0},
{0x010C91, 0x010CD1},
{0x010C92, 0x010CD2},
{0x010C93, 0x010CD3},
{0x010C94, 0x010CD4},
{0x010C95, 0x010CD5},
{0x010C96, 0x010CD6},
{0x010C97, 0x010CD7},
{0x010C98, 0x010CD8},
{0x010C99, 0x010CD9},
{0x010C9A, 0x010CDA},
{0x010C9B, 0x010CDB},
{0x010C9C, 0x010CDC},
{0x010C9D, 0x010CDD},
{0x010C9E, 0x010CDE},
{0x010C9F, 0x010CDF},
{0x010CA0, 0x010CE0},
{0x010CA1, 0x010CE1},
{0x010CA2, 0x010CE2},
{0x010CA3, 0x010CE3},
{0x010CA4, 0x010CE4},
{0x010CA5, 0x010CE5},
{0x010CA6, 0x010CE6},
{0x010CA7, 0x010CE7},
{0x010CA8, 0x010CE8},
{0x010CA9, 0x010CE9},
{0x010CAA, 0x010CEA},
{0x010CAB, 0x010CEB},
{0x010CAC, 0x010CEC},
{0x010CAD, 0x010CED},
{0x010CAE, 0x010CEE},
{0x010CAF, 0x010CEF},
{0x010CB0, 0x010CF0},
{0x010CB1, 0x010CF1},
{0x010CB2, 0x010CF2},
{0x0118A0, 0x0118C0},
{0x0118A1, 0x0118C1},
{0x0118A2, 0x0118C2},
{0x0118A3, 0x0118C3},
{0x0118A4, 0x0118C4},
{0x0118A5, 0x0118C5},
{0x0118A6, 0x0118C6},
{0x0118A7, 0x0118C7},
{0x0118A8, 0x0118C8},
{0x0118A9, 0x0118C9},
{0x0118AA, 0x0118CA},
{0x0118AB, 0x0118CB},
{0x0118AC, 0x0118CC},
{0x0118AD, 0x0118CD},
{0x0118AE, 0x0118CE},
{0x0118AF, 0x0118CF},
{0x0118B0, 0x0118D0},
{0x0118B1, 0x0118D1},
{0x0118B2, 0x0118D2},
{0x0118B3, 0x0118D3},
{0x0118B4, 0x0118D4},
{0x0118B5, 0x0118D5},
{0x0118B6, 0x0118D6},
{0x0118B7, 0x0118D7},
{0x0118B8, 0x0118D8},
{0x0118B9, 0x0118D9},
{0x0118BA, 0x0118DA},
{0x0118BB, 0x0118DB},
{0x0118BC, 0x0118DC},
{0x0118BD, 0x0118DD},
{0x0118BE, 0x0118DE},
{0x0118BF, 0x0118DF},
{0x016E40, 0x016E60},
{0x016E41, 0x016E61},
{0x016E42, 0x016E62},
{0x016E43, 0x016E63},
{0x016E44, 0x016E64},
{0x016E45, 0x016E65},
{0x016E46, 0x016E66},
{0x016E47, 0x016E67},
{0x016E48, 0x016E68},
{0x016E49, 0x016E69},
{0x016E4A, 0x016E6A},
{0x016E4B, 0x016E6B},
{0x016E4C, 0x016E6C},
{0x016E4D, 0x016E6D},
{0x016E4E, 0x016E6E},
{0x016E4F, 0x016E6F},
{0x016E50, 0x016E70},
{0x016E51, 0x016E71},
{0x016E52, 0x016E72},
{0x016E53, 0x016E73},
{0x016E54, 0x016E74},
{0x016E55, 0x016E75},
{0x016E56, 0x016E76},
{0x016E57, 0x016E77},
{0x016E58, 0x016E78},
{0x016E59, 0x016E79},
{0x016E5A, 0x016E7A},
{0x016E5B, 0x016E7B},
{0x016E5C, 0x016E7C},
{0x016E5D, 0x016E7D},
{0x016E5E, 0x016E7E},
{0x016E5F, 0x016E7F},
{0x01E900, 0x01E922},
{0x01E901, 0x01E923},
{0x01E902, 0x01E924},
{0x01E903, 0x01E925},
{0x01E904, 0x01E926},
{0x01E905, 0x01E927},
{0x01E906, 0x01E928},
{0x01E907, 0x01E929},
{0x01E908, 0x01E92A},
{0x01E909, 0x01E92B},
{0x01E90A, 0x01E92C},
{0x01E90B, 0x01E92D},
{0x01E90C, 0x01E92E},
{0x01E90D, 0x01E92F},
{0x01E90E, 0x01E930},
{0x01E90F, 0x01E931},
{0x01E910, 0x01E932},
{0x01E911, 0x01E933},
{0x01E912, 0x01E934},
{0x01E913, 0x01E935},
{0x01E914, 0x01E936},
{0x01E915, 0x01E937},
{0x01E916, 0x01E938},
{0x01E917, 0x01E939},
{0x01E918, 0x01E93A},
{0x01E919, 0x01E93B},
{0x01E91A, 0x01E93C},
{0x01E91B, 0x01E93D},
{0x01E91C, 0x01E93E},
{0x01E91D, 0x01E93F},
{0x01E91E, 0x01E940},
{0x01E91F, 0x01E941},
{0x01E920, 0x01E942},
{0x01E921, 0x01E943},
};

// list is always in ascending order, to enable binary search
const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase = {
{0x000061, 0x000041},
{0x000062, 0x000042},
{0x000063, 0x000043},
{0x000064, 0x000044},
{0x000065, 0x000045},
{0x000066, 0x000046},
{0x000067, 0x000047},
{0x000068, 0x000048},
{0x000069, 0x000049},
{0x00006A, 0x00004A},
{0x00006B, 0x00004B},
{0x00006C, 0x00004C},
{0x00006D, 0x00004D},
{0x00006E, 0x00004E},
{0x00006F, 0x00004F},
{0x000070, 0x000050},
{0x000071, 0x000051},
{0x000072, 0x000052},
{0x000073, 0x000053},
{0x000074, 0x000054},
{0x000075, 0x000055},
{0x000076, 0x000056},
{0x000077, 0x000057},
{0x000078, 0x000058},
{0x000079, 0x000059},
{0x00007A, 0x00005A},
{0x0000B5, 0x00039C},
{0x0000E0, 0x0000C0},
{0x0000E1, 0x0000C1},
{0x0000E2, 0x0000C2},
{0x0000E3, 0x0000C3},
{0x0000E4, 0x0000C4},
{0x0000E5, 0x0000C5},
{0x0000E6, 0x0000C6},
{0x0000E7, 0x0000C7},
{0x0000E8, 0x0000C8},
{0x0000E9, 0x0000C9},
{0x0000EA, 0x0000CA},
{0x0000EB, 0x0000CB},
{0x0000EC, 0x0000CC},
{0x0000ED, 0x0000CD},
{0x0000EE, 0x0000CE},
{0x0000EF, 0x0000CF},
{0x0000F0, 0x0000D0},
{0x0000F1, 0x0000D1},
{0x0000F2, 0x0000D2},
{0x0000F3, 0x0000D3},
{0x0000F4, 0x0000D4},
{0x0000F5, 0x0000D5},
{0x0000F6, 0x0000D6},
{0x0000F8, 0x0000D8},
{0x0000F9, 0x0000D9},
{0x0000FA, 0x0000DA},
{0x0000FB, 0x0000DB},
{0x0000FC, 0x0000DC},
{0x0000FD, 0x0000DD},
{0x0000FE, 0x0000DE},
{0x0000FF, 0x000178},
{0x000101, 0x000100},
{0x000103, 0x000102},
{0x000105, 0x000104},
{0x000107, 0x000106},
{0x000109, 0x000108},
{0x00010B, 0x00010A},
{0x00010D, 0x00010C},
{0x00010F, 0x00010E},
{0x000111, 0x000110},
{0x000113, 0x000112},
{0x000115, 0x000114},
{0x000117, 0x000116},
{0x000119, 0x000118},
{0x00011B, 0x00011A},
{0x00011D, 0x00011C},
{0x00011F, 0x00011E},
{0x000121, 0x000120},
{0x000123, 0x000122},
{0x000125, 0x000124},
{0x000127, 0x000126},
{0x000129, 0x000128},
{0x00012B, 0x00012A},
{0x00012D, 0x00012C},
{0x00012F, 0x00012E},
{0x000131, 0x000049},
{0x000133, 0x000132},
{0x000135, 0x000134},
{0x000137, 0x000136},
{0x00013A, 0x000139},
{0x00013C, 0x00013B},
{0x00013E, 0x00013D},
{0x000140, 0x00013F},
{0x000142, 0x000141},
{0x000144, 0x000143},
{0x000146, 0x000145},
{0x000148, 0x000147},
{0x00014B, 0x00014A},
{0x00014D, 0x00014C},
{0x00014F, 0x00014E},
{0x000151, 0x000150},
{0x000153, 0x000152},
{0x000155, 0x000154},
{0x000157, 0x000156},
{0x000159, 0x000158},
{0x00015B, 0x00015A},
{0x00015D, 0x00015C},
{0x00015F, 0x00015E},
{0x000161, 0x000160},
{0x000163, 0x000162},
{0x000165, 0x000164},
{0x000167, 0x000166},
{0x000169, 0x000168},
{0x00016B, 0x00016A},
{0x00016D, 0x00016C},
{0x00016F, 0x00016E},
{0x000171, 0x000170},
{0x000173, 0x000172},
{0x000175, 0x000174},
{0x000177, 0x000176},
{0x00017A, 0x000179},
{0x00017C, 0x00017B},
{0x00017E, 0x00017D},
{0x00017F, 0x000053},
{0x000180, 0x000243},
{0x000183, 0x000182},
{0x000185, 0x000184},
{0x000188, 0x000187},
{0x00018C, 0x00018B},
{0x000192, 0x000191},
{0x000195, 0x0001F6},
{0x000199, 0x000198},
{0x00019A, 0x00023D},
{0x00019E, 0x000220},
{0x0001A1, 0x0001A0},
{0x0001A3, 0x0001A2},
{0x0001A5, 0x0001A4},
{0x0001A8, 0x0001A7},
{0x0001AD, 0x0001AC},
{0x0001B0, 0x0001AF},
{0x0001B4, 0x0001B3},
{0x0001B6, 0x0001B5},
{0x0001B9, 0x0001B8},
{0x0001BD, 0x0001BC},
{0x0001BF, 0x0001F7},
{0x0001C5, 0x0001C4},
{0x0001C6, 0x0001C4},
{0x0001C8, 0x0001C7},
{0x0001C9, 0x0001C7},
{0x0001CB, 0x0001CA},
{0x0001CC, 0x0001CA},
{0x0001CE, 0x0001CD},
{0x0001D0, 0x0001CF},
{0x0001D2, 0x0001D1},
{0x0001D4, 0x0001D3},
{0x0001D6, 0x0001D5},
{0x0001D8, 0x0001D7},
{0x0001DA, 0x0001D9},
{0x0001DC, 0x0001DB},
{0x0001DD, 0x00018E},
{0x0001DF, 0x0001DE},
{0x0001E1, 0x0001E0},
{0x0001E3, 0x0001E2},
{0x0001E5, 0x0001E4},
{0x0001E7, 0x0001E6},
{0x0001E9, 0x0001E8},
{0x0001EB, 0x0001EA},
{0x0001ED, 0x0001EC},
{0x0001EF, 0x0001EE},
{0x0001F2, 0x0001F1},
{0x0001F3, 0x0001F1},
{0x0001F5, 0x0001F4},
{0x0001F9, 0x0001F8},
{0x0001FB, 0x0001FA},
{0x0001FD, 0x0001FC},
{0x0001FF, 0x0001FE},
{0x000201, 0x000200},
{0x000203, 0x000202},
{0x000205, 0x000204},
{0x000207, 0x000206},
{0x000209, 0x000208},
{0x00020B, 0x00020A},
{0x00020D, 0x00020C},
{0x00020F, 0x00020E},
{0x000211, 0x000210},
{0x000213, 0x000212},
{0x000215, 0x000214},
{0x000217, 0x000216},
{0x000219, 0x000218},
{0x00021B, 0x00021A},
{0x00021D, 0x00021C},
{0x00021F, 0x00021E},
{0x000223, 0x000222},
{0x000225, 0x000224},
{0x000227, 0x000226},
{0x000229, 0x000228},
{0x00022B, 0x00022A},
{0x00022D, 0x00022C},
{0x00022F, 0x00022E},
{0x000231, 0x000230},
{0x000233, 0x000232},
{0x00023C, 0x00023B},
{0x00023F, 0x002C7E},
{0x000240, 0x002C7F},
{0x000242, 0x000241},
{0x000247, 0x000246},
{0x000249, 0x000248},
{0x00024B, 0x00024A},
{0x00024D, 0x00024C},
{0x00024F, 0x00024E},
{0x000250, 0x002C6F},
{0x000251, 0x002C6D},
{0x000252, 0x002C70},
{0x000253, 0x000181},
{0x000254, 0x000186},
{0x000256, 0x000189},
{0x000257, 0x00018A},
{0x000259, 0x00018F},
{0x00025B, 0x000190},
{0x00025C, 0x00A7AB},
{0x000260, 0x000193},
{0x000261, 0x00A7AC},
{0x000263, 0x000194},
{0x000265, 0x00A78D},
{0x000266, 0x00A7AA},
{0x000268, 0x000197},
{0x000269, 0x000196},
{0x00026A, 0x00A7AE},
{0x00026B, 0x002C62},
{0x00026C, 0x00A7AD},
{0x00026F, 0x00019C},
{0x000271, 0x002C6E},
{0x000272, 0x00019D},
{0x000275, 0x00019F},
{0x00027D, 0x002C64},
{0x000280, 0x0001A6},
{0x000282, 0x00A7C5},
{0x000283, 0x0001A9},
{0x000287, 0x00A7B1},
{0x000288, 0x0001AE},
{0x000289, 0x000244},
{0x00028A, 0x0001B1},
{0x00028B, 0x0001B2},
{0x00028C, 0x000245},
{0x000292, 0x0001B7},
{0x00029D, 0x00A7B2},
{0x00029E, 0x00A7B0},
{0x000345, 0x000399},
{0x000371, 0x000370},
{0x000373, 0x000372},
{0x000377, 0x000376},
{0x00037B, 0x0003FD},
{0x00037C, 0x0003FE},
{0x00037D, 0x0003FF},
{0x0003AC, 0x000386},
{0x0003AD, 0x000388},
{0x0003AE, 0x000389},
{0x0003AF, 0x00038A},
{0x0003B1, 0x000391},
{0x0003B2, 0x000392},
{0x0003B3, 0x000393},
{0x0003B4, 0x000394},
{0x0003B5, 0x000395},
{0x0003B6, 0x000396},
{0x0003B7, 0x000397},
{0x0003B8, 0x000398},
{0x0003B9, 0x000399},
{0x0003BA, 0x00039A},
{0x0003BB, 0x00039B},
{0x0003BC, 0x00039C},
{0x0003BD, 0x00039D},
{0x0003BE, 0x00039E},
{0x0003BF, 0x00039F},
{0x0003C0, 0x0003A0},
{0x0003C1, 0x0003A1},
{0x0003C2, 0x0003A3},
{0x0003C3, 0x0003A3},
{0x0003C4, 0x0003A4},
{0x0003C5, 0x0003A5},
{0x0003C6, 0x0003A6},
{0x0003C7, 0x0003A7},
{0x0003C8, 0x0003A8},
{0x0003C9, 0x0003A9},
{0x0003CA, 0x0003AA},
{0x0003CB, 0x0003AB},
{0x0003CC, 0x00038C},
{0x0003CD, 0x00038E},
{0x0003CE, 0x00038F},
{0x0003D0, 0x000392},
{0x0003D1, 0x000398},
{0x0003D5, 0x0003A6},
{0x0003D6, 0x0003A0},
{0x0003D7, 0x0003CF},
{0x0003D9, 0x0003D8},
{0x0003DB, 0x0003DA},
{0x0003DD, 0x0003DC},
{0x0003DF, 0x0003DE},
{0x0003E1, 0x0003E0},
{0x0003E3, 0x0003E2},
{0x0003E5, 0x0003E4},
{0x0003E7, 0x0003E6},
{0x0003E9, 0x0003E8},
{0x0003EB, 0x0003EA},
{0x0003ED, 0x0003EC},
{0x0003EF, 0x0003EE},
{0x0003F0, 0x00039A},
{0x0003F1, 0x0003A1},
{0x0003F2, 0x0003F9},
{0x0003F3, 0x00037F},
{0x0003F5, 0x000395},
{0x0003F8, 0x0003F7},
{0x0003FB, 0x0003FA},
{0x000430, 0x000410},
{0x000431, 0x000411},
{0x000432, 0x000412},
{0x000433, 0x000413},
{0x000434, 0x000414},
{0x000435, 0x000415},
{0x000436, 0x000416},
{0x000437, 0x000417},
{0x000438, 0x000418},
{0x000439, 0x000419},
{0x00043A, 0x00041A},
{0x00043B, 0x00041B},
{0x00043C, 0x00041C},
{0x00043D, 0x00041D},
{0x00043E, 0x00041E},
{0x00043F, 0x00041F},
{0x000440, 0x000420},
{0x000441, 0x000421},
{0x000442, 0x000422},
{0x000443, 0x000423},
{0x000444, 0x000424},
{0x000445, 0x000425},
{0x000446, 0x000426},
{0x000447, 0x000427},
{0x000448, 0x000428},
{0x000449, 0x000429},
{0x00044A, 0x00042A},
{0x00044B, 0x00042B},
{0x00044C, 0x00042C},
{0x00044D, 0x00042D},
{0x00044E, 0x00042E},
{0x00044F, 0x00042F},
{0x000450, 0x000400},
{0x000451, 0x000401},
{0x000452, 0x000402},
{0x000453, 0x000403},
{0x000454, 0x000404},
{0x000455, 0x000405},
{0x000456, 0x000406},
{0x000457, 0x000407},
{0x000458, 0x000408},
{0x000459, 0x000409},
{0x00045A, 0x00040A},
{0x00045B, 0x00040B},
{0x00045C, 0x00040C},
{0x00045D, 0x00040D},
{0x00045E, 0x00040E},
{0x00045F, 0x00040F},
{0x000461, 0x000460},
{0x000463, 0x000462},
{0x000465, 0x000464},
{0x000467, 0x000466},
{0x000469, 0x000468},
{0x00046B, 0x00046A},
{0x00046D, 0x00046C},
{0x00046F, 0x00046E},
{0x000471, 0x000470},
{0x000473, 0x000472},
{0x000475, 0x000474},
{0x000477, 0x000476},
{0x000479, 0x000478},
{0x00047B, 0x00047A},
{0x00047D, 0x00047C},
{0x00047F, 0x00047E},
{0x000481, 0x000480},
{0x00048B, 0x00048A},
{0x00048D, 0x00048C},
{0x00048F, 0x00048E},
{0x000491, 0x000490},
{0x000493, 0x000492},
{0x000495, 0x000494},
{0x000497, 0x000496},
{0x000499, 0x000498},
{0x00049B, 0x00049A},
{0x00049D, 0x00049C},
{0x00049F, 0x00049E},
{0x0004A1, 0x0004A0},
{0x0004A3, 0x0004A2},
{0x0004A5, 0x0004A4},
{0x0004A7, 0x0004A6},
{0x0004A9, 0x0004A8},
{0x0004AB, 0x0004AA},
{0x0004AD, 0x0004AC},
{0x0004AF, 0x0004AE},
{0x0004B1, 0x0004B0},
{0x0004B3, 0x0004B2},
{0x0004B5, 0x0004B4},
{0x0004B7, 0x0004B6},
{0x0004B9, 0x0004B8},
{0x0004BB, 0x0004BA},
{0x0004BD, 0x0004BC},
{0x0004BF, 0x0004BE},
{0x0004C2, 0x0004C1},
{0x0004C4, 0x0004C3},
{0x0004C6, 0x0004C5},
{0x0004C8, 0x0004C7},
{0x0004CA, 0x0004C9},
{0x0004CC, 0x0004CB},
{0x0004CE, 0x0004CD},
{0x0004CF, 0x0004C0},
{0x0004D1, 0x0004D0},
{0x0004D3, 0x0004D2},
{0x0004D5, 0x0004D4},
{0x0004D7, 0x0004D6},
{0x0004D9, 0x0004D8},
{0x0004DB, 0x0004DA},
{0x0004DD, 0x0004DC},
{0x0004DF, 0x0004DE},
{0x0004E1, 0x0004E0},
{0x0004E3, 0x0004E2},
{0x0004E5, 0x0004E4},
{0x0004E7, 0x0004E6},
{0x0004E9, 0x0004E8},
{0x0004EB, 0x0004EA},
{0x0004ED, 0x0004EC},
{0x0004EF, 0x0004EE},
{0x0004F1, 0x0004F0},
{0x0004F3, 0x0004F2},
{0x0004F5, 0x0004F4},
{0x0004F7, 0x0004F6},
{0x0004F9, 0x0004F8},
{0x0004FB, 0x0004FA},
{0x0004FD, 0x0004FC},
{0x0004FF, 0x0004FE},
{0x000501, 0x000500},
{0x000503, 0x000502},
{0x000505, 0x000504},
{0x000507, 0x000506},
{0x000509, 0x000508},
{0x00050B, 0x00050A},
{0x00050D, 0x00050C},
{0x00050F, 0x00050E},
{0x000511, 0x000510},
{0x000513, 0x000512},
{0x000515, 0x000514},
{0x000517, 0x000516},
{0x000519, 0x000518},
{0x00051B, 0x00051A},
{0x00051D, 0x00051C},
{0x00051F, 0x00051E},
{0x000521, 0x000520},
{0x000523, 0x000522},
{0x000525, 0x000524},
{0x000527, 0x000526},
{0x000529, 0x000528},
{0x00052B, 0x00052A},
{0x00052D, 0x00052C},
{0x00052F, 0x00052E},
{0x000561, 0x000531},
{0x000562, 0x000532},
{0x000563, 0x000533},
{0x000564, 0x000534},
{0x000565, 0x000535},
{0x000566, 0x000536},
{0x000567, 0x000537},
{0x000568, 0x000538},
{0x000569, 0x000539},
{0x00056A, 0x00053A},
{0x00056B, 0x00053B},
{0x00056C, 0x00053C},
{0x00056D, 0x00053D},
{0x00056E, 0x00053E},
{0x00056F, 0x00053F},
{0x000570, 0x000540},
{0x000571, 0x000541},
{0x000572, 0x000542},
{0x000573, 0x000543},
{0x000574, 0x000544},
{0x000575, 0x000545},
{0x000576, 0x000546},
{0x000577, 0x000547},
{0x000578, 0x000548},
{0x000579, 0x000549},
{0x00057A, 0x00054A},
{0x00057B, 0x00054B},
{0x00057C, 0x00054C},
{0x00057D, 0x00054D},
{0x00057E, 0x00054E},
{0x00057F, 0x00054F},
{0x000580, 0x000550},
{0x000581, 0x000551},
{0x000582, 0x000552},
{0x000583, 0x000553},
{0x000584, 0x000554},
{0x000585, 0x000555},
{0x000586, 0x000556},
{0x0010D0, 0x001C90},
{0x0010D1, 0x001C91},
{0x0010D2, 0x001C92},
{0x0010D3, 0x001C93},
{0x0010D4, 0x001C94},
{0x0010D5, 0x001C95},
{0x0010D6, 0x001C96},
{0x0010D7, 0x001C97},
{0x0010D8, 0x001C98},
{0x0010D9, 0x001C99},
{0x0010DA, 0x001C9A},
{0x0010DB, 0x001C9B},
{0x0010DC, 0x001C9C},
{0x0010DD, 0x001C9D},
{0x0010DE, 0x001C9E},
{0x0010DF, 0x001C9F},
{0x0010E0, 0x001CA0},
{0x0010E1, 0x001CA1},
{0x0010E2, 0x001CA2},
{0x0010E3, 0x001CA3},
{0x0010E4, 0x001CA4},
{0x0010E5, 0x001CA5},
{0x0010E6, 0x001CA6},
{0x0010E7, 0x001CA7},
{0x0010E8, 0x001CA8},
{0x0010E9, 0x001CA9},
{0x0010EA, 0x001CAA},
{0x0010EB, 0x001CAB},
{0x0010EC, 0x001CAC},
{0x0010ED, 0x001CAD},
{0x0010EE, 0x001CAE},
{0x0010EF, 0x001CAF},
{0x0010F0, 0x001CB0},
{0x0010F1, 0x001CB1},
{0x0010F2, 0x001CB2},
{0x0010F3, 0x001CB3},
{0x0010F4, 0x001CB4},
{0x0010F5, 0x001CB5},
{0x0010F6, 0x001CB6},
{0x0010F7, 0x001CB7},
{0x0010F8, 0x001CB8},
{0x0010F9, 0x001CB9},
{0x0010FA, 0x001CBA},
{0x0010FD, 0x001CBD},
{0x0010FE, 0x001CBE},
{0x0010FF, 0x001CBF},
{0x0013F8, 0x0013F0},
{0x0013F9, 0x0013F1},
{0x0013FA, 0x0013F2},
{0x0013FB, 0x0013F3},
{0x0013FC, 0x0013F4},
{0x0013FD, 0x0013F5},
{0x001C80, 0x000412},
{0x001C81, 0x000414},
{0x001C82, 0x00041E},
{0x001C83, 0x000421},
{0x001C84, 0x000422},
{0x001C85, 0x000422},
{0x001C86, 0x00042A},
{0x001C87, 0x000462},
{0x001C88, 0x00A64A},
{0x001D79, 0x00A77D},
{0x001D7D, 0x002C63},
{0x001D8E, 0x00A7C6},
{0x001E01, 0x001E00},
{0x001E03, 0x001E02},
{0x001E05, 0x001E04},
{0x001E07, 0x001E06},
{0x001E09, 0x001E08},
{0x001E0B, 0x001E0A},
{0x001E0D, 0x001E0C},
{0x001E0F, 0x001E0E},
{0x001E11, 0x001E10},
{0x001E13, 0x001E12},
{0x001E15, 0x001E14},
{0x001E17, 0x001E16},
{0x001E19, 0x001E18},
{0x001E1B, 0x001E1A},
{0x001E1D, 0x001E1C},
{0x001E1F, 0x001E1E},
{0x001E21, 0x001E20},
{0x001E23, 0x001E22},
{0x001E25, 0x001E24},
{0x001E27, 0x001E26},
{0x001E29, 0x001E28},
{0x001E2B, 0x001E2A},
{0x001E2D, 0x001E2C},
{0x001E2F, 0x001E2E},
{0x001E31, 0x001E30},
{0x001E33, 0x001E32},
{0x001E35, 0x001E34},
{0x001E37, 0x001E36},
{0x001E39, 0x001E38},
{0x001E3B, 0x001E3A},
{0x001E3D, 0x001E3C},
{0x001E3F, 0x001E3E},
{0x001E41, 0x001E40},
{0x001E43, 0x001E42},
{0x001E45, 0x001E44},
{0x001E47, 0x001E46},
{0x001E49, 0x001E48},
{0x001E4B, 0x001E4A},
{0x001E4D, 0x001E4C},
{0x001E4F, 0x001E4E},
{0x001E51, 0x001E50},
{0x001E53, 0x001E52},
{0x001E55, 0x001E54},
{0x001E57, 0x001E56},
{0x001E59, 0x001E58},
{0x001E5B, 0x001E5A},
{0x001E5D, 0x001E5C},
{0x001E5F, 0x001E5E},
{0x001E61, 0x001E60},
{0x001E63, 0x001E62},
{0x001E65, 0x001E64},
{0x001E67, 0x001E66},
{0x001E69, 0x001E68},
{0x001E6B, 0x001E6A},
{0x001E6D, 0x001E6C},
{0x001E6F, 0x001E6E},
{0x001E71, 0x001E70},
{0x001E73, 0x001E72},
{0x001E75, 0x001E74},
{0x001E77, 0x001E76},
{0x001E79, 0x001E78},
{0x001E7B, 0x001E7A},
{0x001E7D, 0x001E7C},
{0x001E7F, 0x001E7E},
{0x001E81, 0x001E80},
{0x001E83, 0x001E82},
{0x001E85, 0x001E84},
{0x001E87, 0x001E86},
{0x001E89, 0x001E88},
{0x001E8B, 0x001E8A},
{0x001E8D, 0x001E8C},
{0x001E8F, 0x001E8E},
{0x001E91, 0x001E90},
{0x001E93, 0x001E92},
{0x001E95, 0x001E94},
{0x001E9B, 0x001E60},
{0x001EA1, 0x001EA0},
{0x001EA3, 0x001EA2},
{0x001EA5, 0x001EA4},
{0x001EA7, 0x001EA6},
{0x001EA9, 0x001EA8},
{0x001EAB, 0x001EAA},
{0x001EAD, 0x001EAC},
{0x001EAF, 0x001EAE},
{0x001EB1, 0x001EB0},
{0x001EB3, 0x001EB2},
{0x001EB5, 0x001EB4},
{0x001EB7, 0x001EB6},
{0x001EB9, 0x001EB8},
{0x001EBB, 0x001EBA},
{0x001EBD, 0x001EBC},
{0x001EBF, 0x001EBE},
{0x001EC1, 0x001EC0},
{0x001EC3, 0x001EC2},
{0x001EC5, 0x001EC4},
{0x001EC7, 0x001EC6},
{0x001EC9, 0x001EC8},
{0x001ECB, 0x001ECA},
{0x001ECD, 0x001ECC},
{0x001ECF, 0x001ECE},
{0x001ED1, 0x001ED0},
{0x001ED3, 0x001ED2},
{0x001ED5, 0x001ED4},
{0x001ED7, 0x001ED6},
{0x001ED9, 0x001ED8},
{0x001EDB, 0x001EDA},
{0x001EDD, 0x001EDC},
{0x001EDF, 0x001EDE},
{0x001EE1, 0x001EE0},
{0x001EE3, 0x001EE2},
{0x001EE5, 0x001EE4},
{0x001EE7, 0x001EE6},
{0x001EE9, 0x001EE8},
{0x001EEB, 0x001EEA},
{0x001EED, 0x001EEC},
{0x001EEF, 0x001EEE},
{0x001EF1, 0x001EF0},
{0x001EF3, 0x001EF2},
{0x001EF5, 0x001EF4},
{0x001EF7, 0x001EF6},
{0x001EF9, 0x001EF8},
{0x001EFB, 0x001EFA},
{0x001EFD, 0x001EFC},
{0x001EFF, 0x001EFE},
{0x001F00, 0x001F08},
{0x001F01, 0x001F09},
{0x001F02, 0x001F0A},
{0x001F03, 0x001F0B},
{0x001F04, 0x001F0C},
{0x001F05, 0x001F0D},
{0x001F06, 0x001F0E},
{0x001F07, 0x001F0F},
{0x001F10, 0x001F18},
{0x001F11, 0x001F19},
{0x001F12, 0x001F1A},
{0x001F13, 0x001F1B},
{0x001F14, 0x001F1C},
{0x001F15, 0x001F1D},
{0x001F20, 0x001F28},
{0x001F21, 0x001F29},
{0x001F22, 0x001F2A},
{0x001F23, 0x001F2B},
{0x001F24, 0x001F2C},
{0x001F25, 0x001F2D},
{0x001F26, 0x001F2E},
{0x001F27, 0x001F2F},
{0x001F30, 0x001F38},
{0x001F31, 0x001F39},
{0x001F32, 0x001F3A},
{0x001F33, 0x001F3B},
{0x001F34, 0x001F3C},
{0x001F35, 0x001F3D},
{0x001F36, 0x001F3E},
{0x001F37, 0x001F3F},
{0x001F40, 0x001F48},
{0x001F41, 0x001F49},
{0x001F42, 0x001F4A},
{0x001F43, 0x001F4B},
{0x001F44, 0x001F4C},
{0x001F45, 0x001F4D},
{0x001F51, 0x001F59},
{0x001F53, 0x001F5B},
{0x001F55, 0x001F5D},
{0x001F57, 0x001F5F},
{0x001F60, 0x001F68},
{0x001F61, 0x001F69},
{0x001F62, 0x001F6A},
{0x001F63, 0x001F6B},
{0x001F64, 0x001F6C},
{0x001F65, 0x001F6D},
{0x001F66, 0x001F6E},
{0x001F67, 0x001F6F},
{0x001F70, 0x001FBA},
{0x001F71, 0x001FBB},
{0x001F72, 0x001FC8},
{0x001F73, 0x001FC9},
{0x001F74, 0x001FCA},
{0x001F75, 0x001FCB},
{0x001F76, 0x001FDA},
{0x001F77, 0x001FDB},
{0x001F78, 0x001FF8},
{0x001F79, 0x001FF9},
{0x001F7A, 0x001FEA},
{0x001F7B, 0x001FEB},
{0x001F7C, 0x001FFA},
{0x001F7D, 0x001FFB},
{0x001F80, 0x001F88},
{0x001F81, 0x001F89},
{0x001F82, 0x001F8A},
{0x001F83, 0x001F8B},
{0x001F84, 0x001F8C},
{0x001F85, 0x001F8D},
{0x001F86, 0x001F8E},
{0x001F87, 0x001F8F},
{0x001F90, 0x001F98},
{0x001F91, 0x001F99},
{0x001F92, 0x001F9A},
{0x001F93, 0x001F9B},
{0x001F94, 0x001F9C},
{0x001F95, 0x001F9D},
{0x001F96, 0x001F9E},
{0x001F97, 0x001F9F},
{0x001FA0, 0x001FA8},
{0x001FA1, 0x001FA9},
{0x001FA2, 0x001FAA},
{0x001FA3, 0x001FAB},
{0x001FA4, 0x001FAC},
{0x001FA5, 0x001FAD},
{0x001FA6, 0x001FAE},
{0x001FA7, 0x001FAF},
{0x001FB0, 0x001FB8},
{0x001FB1, 0x001FB9},
{0x001FB3, 0x001FBC},
{0x001FBE, 0x000399},
{0x001FC3, 0x001FCC},
{0x001FD0, 0x001FD8},
{0x001FD1, 0x001FD9},
{0x001FE0, 0x001FE8},
{0x001FE1, 0x001FE9},
{0x001FE5, 0x001FEC},
{0x001FF3, 0x001FFC},
{0x00214E, 0x002132},
{0x002170, 0x002160},
{0x002171, 0x002161},
{0x002172, 0x002162},
{0x002173, 0x002163},
{0x002174, 0x002164},
{0x002175, 0x002165},
{0x002176, 0x002166},
{0x002177, 0x002167},
{0x002178, 0x002168},
{0x002179, 0x002169},
{0x00217A, 0x00216A},
{0x00217B, 0x00216B},
{0x00217C, 0x00216C},
{0x00217D, 0x00216D},
{0x00217E, 0x00216E},
{0x00217F, 0x00216F},
{0x002184, 0x002183},
{0x0024D0, 0x0024B6},
{0x0024D1, 0x0024B7},
{0x0024D2, 0x0024B8},
{0x0024D3, 0x0024B9},
{0x0024D4, 0x0024BA},
{0x0024D5, 0x0024BB},
{0x0024D6, 0x0024BC},
{0x0024D7, 0x0024BD},
{0x0024D8, 0x0024BE},
{0x0024D9, 0x0024BF},
{0x0024DA, 0x0024C0},
{0x0024DB, 0x0024C1},
{0x0024DC, 0x0024C2},
{0x0024DD, 0x0024C3},
{0x0024DE, 0x0024C4},
{0x0024DF, 0x0024C5},
{0x0024E0, 0x0024C6},
{0x0024E1, 0x0024C7},
{0x0024E2, 0x0024C8},
{0x0024E3, 0x0024C9},
{0x0024E4, 0x0024CA},
{0x0024E5, 0x0024CB},
{0x0024E6, 0x0024CC},
{0x0024E7, 0x0024CD},
{0x0024E8, 0x0024CE},
{0x0024E9, 0x0024CF},
{0x002C30, 0x002C00},
{0x002C31, 0x002C01},
{0x002C32, 0x002C02},
{0x002C33, 0x002C03},
{0x002C34, 0x002C04},
{0x002C35, 0x002C05},
{0x002C36, 0x002C06},
{0x002C37, 0x002C07},
{0x002C38, 0x002C08},
{0x002C39, 0x002C09},
{0x002C3A, 0x002C0A},
{0x002C3B, 0x002C0B},
{0x002C3C, 0x002C0C},
{0x002C3D, 0x002C0D},
{0x002C3E, 0x002C0E},
{0x002C3F, 0x002C0F},
{0x002C40, 0x002C10},
{0x002C41, 0x002C11},
{0x002C42, 0x002C12},
{0x002C43, 0x002C13},
{0x002C44, 0x002C14},
{0x002C45, 0x002C15},
{0x002C46, 0x002C16},
{0x002C47, 0x002C17},
{0x002C48, 0x002C18},
{0x002C49, 0x002C19},
{0x002C4A, 0x002C1A},
{0x002C4B, 0x002C1B},
{0x002C4C, 0x002C1C},
{0x002C4D, 0x002C1D},
{0x002C4E, 0x002C1E},
{0x002C4F, 0x002C1F},
{0x002C50, 0x002C20},
{0x002C51, 0x002C21},
{0x002C52, 0x002C22},
{0x002C53, 0x002C23},
{0x002C54, 0x002C24},
{0x002C55, 0x002C25},
{0x002C56, 0x002C26},
{0x002C57, 0x002C27},
{0x002C58, 0x002C28},
{0x002C59, 0x002C29},
{0x002C5A, 0x002C2A},
{0x002C5B, 0x002C2B},
{0x002C5C, 0x002C2C},
{0x002C5D, 0x002C2D},
{0x002C5E, 0x002C2E},
{0x002C5F, 0x002C2F},
{0x002C61, 0x002C60},
{0x002C65, 0x00023A},
{0x002C66, 0x00023E},
{0x002C68, 0x002C67},
{0x002C6A, 0x002C69},
{0x002C6C, 0x002C6B},
{0x002C73, 0x002C72},
{0x002C76, 0x002C75},
{0x002C81, 0x002C80},
{0x002C83, 0x002C82},
{0x002C85, 0x002C84},
{0x002C87, 0x002C86},
{0x002C89, 0x002C88},
{0x002C8B, 0x002C8A},
{0x002C8D, 0x002C8C},
{0x002C8F, 0x002C8E},
{0x002C91, 0x002C90},
{0x002C93, 0x002C92},
{0x002C95, 0x002C94},
{0x002C97, 0x002C96},
{0x002C99, 0x002C98},
{0x002C9B, 0x002C9A},
{0x002C9D, 0x002C9C},
{0x002C9F, 0x002C9E},
{0x002CA1, 0x002CA0},
{0x002CA3, 0x002CA2},
{0x002CA5, 0x002CA4},
{0x002CA7, 0x002CA6},
{0x002CA9, 0x002CA8},
{0x002CAB, 0x002CAA},
{0x002CAD, 0x002CAC},
{0x002CAF, 0x002CAE},
{0x002CB1, 0x002CB0},
{0x002CB3, 0x002CB2},
{0x002CB5, 0x002CB4},
{0x002CB7, 0x002CB6},
{0x002CB9, 0x002CB8},
{0x002CBB, 0x002CBA},
{0x002CBD, 0x002CBC},
{0x002CBF, 0x002CBE},
{0x002CC1, 0x002CC0},
{0x002CC3, 0x002CC2},
{0x002CC5, 0x002CC4},
{0x002CC7, 0x002CC6},
{0x002CC9, 0x002CC8},
{0x002CCB, 0x002CCA},
{0x002CCD, 0x002CCC},
{0x002CCF, 0x002CCE},
{0x002CD1, 0x002CD0},
{0x002CD3, 0x002CD2},
{0x002CD5, 0x002CD4},
{0x002CD7, 0x002CD6},
{0x002CD9, 0x002CD8},
{0x002CDB, 0x002CDA},
{0x002CDD, 0x002CDC},
{0x002CDF, 0x002CDE},
{0x002CE1, 0x002CE0},
{0x002CE3, 0x002CE2},
{0x002CEC, 0x002CEB},
{0x002CEE, 0x002CED},
{0x002CF3, 0x002CF2},
{0x002D00, 0x0010A0},
{0x002D01, 0x0010A1},
{0x002D02, 0x0010A2},
{0x002D03, 0x0010A3},
{0x002D04, 0x0010A4},
{0x002D05, 0x0010A5},
{0x002D06, 0x0010A6},
{0x002D07, 0x0010A7},
{0x002D08, 0x0010A8},
{0x002D09, 0x0010A9},
{0x002D0A, 0x0010AA},
{0x002D0B, 0x0010AB},
{0x002D0C, 0x0010AC},
{0x002D0D, 0x0010AD},
{0x002D0E, 0x0010AE},
{0x002D0F, 0x0010AF},
{0x002D10, 0x0010B0},
{0x002D11, 0x0010B1},
{0x002D12, 0x0010B2},
{0x002D13, 0x0010B3},
{0x002D14, 0x0010B4},
{0x002D15, 0x0010B5},
{0x002D16, 0x0010B6},
{0x002D17, 0x0010B7},
{0x002D18, 0x0010B8},
{0x002D19, 0x0010B9},
{0x002D1A, 0x0010BA},
{0x002D1B, 0x0010BB},
{0x002D1C, 0x0010BC},
{0x002D1D, 0x0010BD},
{0x002D1E, 0x0010BE},
{0x002D1F, 0x0010BF},
{0x002D20, 0x0010C0},
{0x002D21, 0x0010C1},
{0x002D22, 0x0010C2},
{0x002D23, 0x0010C3},
{0x002D24, 0x0010C4},
{0x002D25, 0x0010C5},
{0x002D27, 0x0010C7},
{0x002D2D, 0x0010CD},
{0x00A641, 0x00A640},
{0x00A643, 0x00A642},
{0x00A645, 0x00A644},
{0x00A647, 0x00A646},
{0x00A649, 0x00A648},
{0x00A64B, 0x00A64A},
{0x00A64D, 0x00A64C},
{0x00A64F, 0x00A64E},
{0x00A651, 0x00A650},
{0x00A653, 0x00A652},
{0x00A655, 0x00A654},
{0x00A657, 0x00A656},
{0x00A659, 0x00A658},
{0x00A65B, 0x00A65A},
{0x00A65D, 0x00A65C},
{0x00A65F, 0x00A65E},
{0x00A661, 0x00A660},
{0x00A663, 0x00A662},
{0x00A665, 0x00A664},
{0x00A667, 0x00A666},
{0x00A669, 0x00A668},
{0x00A66B, 0x00A66A},
{0x00A66D, 0x00A66C},
{0x00A681, 0x00A680},
{0x00A683, 0x00A682},
{0x00A685, 0x00A684},
{0x00A687, 0x00A686},
{0x00A689, 0x00A688},
{0x00A68B, 0x00A68A},
{0x00A68D, 0x00A68C},
{0x00A68F, 0x00A68E},
{0x00A691, 0x00A690},
{0x00A693, 0x00A692},
{0x00A695, 0x00A694},
{0x00A697, 0x00A696},
{0x00A699, 0x00A698},
{0x00A69B, 0x00A69A},
{0x00A723, 0x00A722},
{0x00A725, 0x00A724},
{0x00A727, 0x00A726},
{0x00A729, 0x00A728},
{0x00A72B, 0x00A72A},
{0x00A72D, 0x00A72C},
{0x00A72F, 0x00A72E},
{0x00A733, 0x00A732},
{0x00A735, 0x00A734},
{0x00A737, 0x00A736},
{0x00A739, 0x00A738},
{0x00A73B, 0x00A73A},
{0x00A73D, 0x00A73C},
{0x00A73F, 0x00A73E},
{0x00A741, 0x00A740},
{0x00A743, 0x00A742},
{0x00A745, 0x00A744},
{0x00A747, 0x00A746},
{0x00A749, 0x00A748},
{0x00A74B, 0x00A74A},
{0x00A74D, 0x00A74C},
{0x00A74F, 0x00A74E},
{0x00A751, 0x00A750},
{0x00A753, 0x00A752},
{0x00A755, 0x00A754},
{0x00A757, 0x00A756},
{0x00A759, 0x00A758},
{0x00A75B, 0x00A75A},
{0x00A75D, 0x00A75C},
{0x00A75F, 0x00A75E},
{0x00A761, 0x00A760},
{0x00A763, 0x00A762},
{0x00A765, 0x00A764},
{0x00A767, 0x00A766},
{0x00A769, 0x00A768},
{0x00A76B, 0x00A76A},
{0x00A76D, 0x00A76C},
{0x00A76F, 0x00A76E},
{0x00A77A, 0x00A779},
{0x00A77C, 0x00A77B},
{0x00A77F, 0x00A77E},
{0x00A781, 0x00A780},
{0x00A783, 0x00A782},
{0x00A785, 0x00A784},
{0x00A787, 0x00A786},
{0x00A78C, 0x00A78B},
{0x00A791, 0x00A790},
{0x00A793, 0x00A792},
{0x00A794, 0x00A7C4},
{0x00A797, 0x00A796},
{0x00A799, 0x00A798},
{0x00A79B, 0x00A79A},
{0x00A79D, 0x00A79C},
{0x00A79F, 0x00A79E},
{0x00A7A1, 0x00A7A0},
{0x00A7A3, 0x00A7A2},
{0x00A7A5, 0x00A7A4},
{0x00A7A7, 0x00A7A6},
{0x00A7A9, 0x00A7A8},
{0x00A7B5, 0x00A7B4},
{0x00A7B7, 0x00A7B6},
{0x00A7B9, 0x00A7B8},
{0x00A7BB, 0x00A7BA},
{0x00A7BD, 0x00A7BC},
{0x00A7BF, 0x00A7BE},
{0x00A7C1, 0x00A7C0},
{0x00A7C3, 0x00A7C2},
{0x00A7C8, 0x00A7C7},
{0x00A7CA, 0x00A7C9},
{0x00A7D1, 0x00A7D0},
{0x00A7D7, 0x00A7D6},
{0x00A7D9, 0x00A7D8},
{0x00A7F6, 0x00A7F5},
{0x00AB53, 0x00A7B3},
{0x00AB70, 0x0013A0},
{0x00AB71, 0x0013A1},
{0x00AB72, 0x0013A2},
{0x00AB73, 0x0013A3},
{0x00AB74, 0x0013A4},
{0x00AB75, 0x0013A5},
{0x00AB76, 0x0013A6},
{0x00AB77, 0x0013A7},
{0x00AB78, 0x0013A8},
{0x00AB79, 0x0013A9},
{0x00AB7A, 0x0013AA},
{0x00AB7B, 0x0013AB},
{0x00AB7C, 0x0013AC},
{0x00AB7D, 0x0013AD},
{0x00AB7E, 0x0013AE},
{0x00AB7F, 0x0013AF},
{0x00AB80, 0x0013B0},
{0x00AB81, 0x0013B1},
{0x00AB82, 0x0013B2},
{0x00AB83, 0x0013B3},
{0x00AB84, 0x0013B4},
{0x00AB85, 0x0013B5},
{0x00AB86, 0x0013B6},
{0x00AB87, 0x0013B7},
{0x00AB88, 0x0013B8},
{0x00AB89, 0x0013B9},
{0x00AB8A, 0x0013BA},
{0x00AB8B, 0x0013BB},
{0x00AB8C, 0x0013BC},
{0x00AB8D, 0x0013BD},
{0x00AB8E, 0x0013BE},
{0x00AB8F, 0x0013BF},
{0x00AB90, 0x0013C0},
{0x00AB91, 0x0013C1},
{0x00AB92, 0x0013C2},
{0x00AB93, 0x0013C3},
{0x00AB94, 0x0013C4},
{0x00AB95, 0x0013C5},
{0x00AB96, 0x0013C6},
{0x00AB97, 0x0013C7},
{0x00AB98, 0x0013C8},
{0x00AB99, 0x0013C9},
{0x00AB9A, 0x0013CA},
{0x00AB9B, 0x0013CB},
{0x00AB9C, 0x0013CC},
{0x00AB9D, 0x0013CD},
{0x00AB9E, 0x0013CE},
{0x00AB9F, 0x0013CF},
{0x00ABA0, 0x0013D0},
{0x00ABA1, 0x0013D1},
{0x00ABA2, 0x0013D2},
{0x00ABA3, 0x0013D3},
{0x00ABA4, 0x0013D4},
{0x00ABA5, 0x0013D5},
{0x00ABA6, 0x0013D6},
{0x00ABA7, 0x0013D7},
{0x00ABA8, 0x0013D8},
{0x00ABA9, 0x0013D9},
{0x00ABAA, 0x0013DA},
{0x00ABAB, 0x0013DB},
{0x00ABAC, 0x0013DC},
{0x00ABAD, 0x0013DD},
{0x00ABAE, 0x0013DE},
{0x00ABAF, 0x0013DF},
{0x00ABB0, 0x0013E0},
{0x00ABB1, 0x0013E1},
{0x00ABB2, 0x0013E2},
{0x00ABB3, 0x0013E3},
{0x00ABB4, 0x0013E4},
{0x00ABB5, 0x0013E5},
{0x00ABB6, 0x0013E6},
{0x00ABB7, 0x0013E7},
{0x00ABB8, 0x0013E8},
{0x00ABB9, 0x0013E9},
{0x00ABBA, 0x0013EA},
{0x00ABBB, 0x0013EB},
{0x00ABBC, 0x0013EC},
{0x00ABBD, 0x0013ED},
{0x00ABBE, 0x0013EE},
{0x00ABBF, 0x0013EF},
{0x00FF41, 0x00FF21},
{0x00FF42, 0x00FF22},
{0x00FF43, 0x00FF23},
{0x00FF44, 0x00FF24},
{0x00FF45, 0x00FF25},
{0x00FF46, 0x00FF26},
{0x00FF47, 0x00FF27},
{0x00FF48, 0x00FF28},
{0x00FF49, 0x00FF29},
{0x00FF4A, 0x00FF2A},
{0x00FF4B, 0x00FF2B},
{0x00FF4C, 0x00FF2C},
{0x00FF4D, 0x00FF2D},
{0x00FF4E, 0x00FF2E},
{0x00FF4F, 0x00FF2F},
{0x00FF50, 0x00FF30},
{0x00FF51, 0x00FF31},
{0x00FF52, 0x00FF32},
{0x00FF53, 0x00FF33},
{0x00FF54, 0x00FF34},
{0x00FF55, 0x00FF35},
{0x00FF56, 0x00FF36},
{0x00FF57, 0x00FF37},
{0x00FF58, 0x00FF38},
{0x00FF59, 0x00FF39},
{0x00FF5A, 0x00FF3A},
{0x010428, 0x010400},
{0x010429, 0x010401},
{0x01042A, 0x010402},
{0x01042B, 0x010403},
{0x01042C, 0x010404},
{0x01042D, 0x010405},
{0x01042E, 0x010406},
{0x01042F, 0x010407},
{0x010430, 0x010408},
{0x010431, 0x010409},
{0x010432, 0x01040A},
{0x010433, 0x01040B},
{0x010434, 0x01040C},
{0x010435, 0x01040D},
{0x010436, 0x01040E},
{0x010437, 0x01040F},
{0x010438, 0x010410},
{0x010439, 0x010411},
{0x01043A, 0x010412},
{0x01043B, 0x010413},
{0x01043C, 0x010414},
{0x01043D, 0x010415},
{0x01043E, 0x010416},
{0x01043F, 0x010417},
{0x010440, 0x010418},
{0x010441, 0x010419},
{0x010442, 0x01041A},
{0x010443, 0x01041B},
{0x010444, 0x01041C},
{0x010445, 0x01041D},
{0x010446, 0x01041E},
{0x010447, 0x01041F},
{0x010448, 0x010420},
{0x010449, 0x010421},
{0x01044A, 0x010422},
{0x01044B, 0x010423},
{0x01044C, 0x010424},
{0x01044D, 0x010425},
{0x01044E, 0x010426},
{0x01044F, 0x010427},
{0x0104D8, 0x0104B0},
{0x0104D9, 0x0104B1},
{0x0104DA, 0x0104B2},
{0x0104DB, 0x0104B3},
{0x0104DC, 0x0104B4},
{0x0104DD, 0x0104B5},
{0x0104DE, 0x0104B6},
{0x0104DF, 0x0104B7},
{0x0104E0, 0x0104B8},
{0x0104E1, 0x0104B9},
{0x0104E2, 0x0104BA},
{0x0104E3, 0x0104BB},
{0x0104E4, 0x0104BC},
{0x0104E5, 0x0104BD},
{0x0104E6, 0x0104BE},
{0x0104E7, 0x0104BF},
{0x0104E8, 0x0104C0},
{0x0104E9, 0x0104C1},
{0x0104EA, 0x0104C2},
{0x0104EB, 0x0104C3},
{0x0104EC, 0x0104C4},
{0x0104ED, 0x0104C5},
{0x0104EE, 0x0104C6},
{0x0104EF, 0x0104C7},
{0x0104F0, 0x0104C8},
{0x0104F1, 0x0104C9},
{0x0104F2, 0x0104CA},
{0x0104F3, 0x0104CB},
{0x0104F4, 0x0104CC},
{0x0104F5, 0x0104CD},
{0x0104F6, 0x0104CE},
{0x0104F7, 0x0104CF},
{0x0104F8, 0x0104D0},
{0x0104F9, 0x0104D1},
{0x0104FA, 0x0104D2},
{0x0104FB, 0x0104D3},
{0x010597, 0x010570},
{0x010598, 0x010571},
{0x010599, 0x010572},
{0x01059A, 0x010573},
{0x01059B, 0x010574},
{0x01059C, 0x010575},
{0x01059D, 0x010576},
{0x01059E, 0x010577},
{0x01059F, 0x010578},
{0x0105A0, 0x010579},
{0x0105A1, 0x01057A},
{0x0105A3, 0x01057C},
{0x0105A4, 0x01057D},
{0x0105A5, 0x01057E},
{0x0105A6, 0x01057F},
{0x0105A7, 0x010580},
{0x0105A8, 0x010581},
{0x0105A9, 0x010582},
{0x0105AA, 0x010583},
{0x0105AB, 0x010584},
{0x0105AC, 0x010585},
{0x0105AD, 0x010586},
{0x0105AE, 0x010587},
{0x0105AF, 0x010588},
{0x0105B0, 0x010589},
{0x0105B1, 0x01058A},
{0x0105B3, 0x01058C},
{0x0105B4, 0x01058D},
{0x0105B5, 0x01058E},
{0x0105B6, 0x01058F},
{0x0105B7, 0x010590},
{0x0105B8, 0x010591},
{0x0105B9, 0x010592},
{0x0105BB, 0x010594},
{0x0105BC, 0x010595},
{0x010CC0, 0x010C80},
{0x010CC1, 0x010C81},
{0x010CC2, 0x010C82},
{0x010CC3, 0x010C83},
{0x010CC4, 0x010C84},
{0x010CC5, 0x010C85},
{0x010CC6, 0x010C86},
{0x010CC7, 0x010C87},
{0x010CC8, 0x010C88},
{0x010CC9, 0x010C89},
{0x010CCA, 0x010C8A},
{0x010CCB, 0x010C8B},
{0x010CCC, 0x010C8C},
{0x010CCD, 0x010C8D},
{0x010CCE, 0x010C8E},
{0x010CCF, 0x010C8F},
{0x010CD0, 0x010C90},
{0x010CD1, 0x010C91},
{0x010CD2, 0x010C92},
{0x010CD3, 0x010C93},
{0x010CD4, 0x010C94},
{0x010CD5, 0x010C95},
{0x010CD6, 0x010C96},
{0x010CD7, 0x010C97},
{0x010CD8, 0x010C98},
{0x010CD9, 0x010C99},
{0x010CDA, 0x010C9A},
{0x010CDB, 0x010C9B},
{0x010CDC, 0x010C9C},
{0x010CDD, 0x010C9D},
{0x010CDE, 0x010C9E},
{0x010CDF, 0x010C9F},
{0x010CE0, 0x010CA0},
{0x010CE1, 0x010CA1},
{0x010CE2, 0x010CA2},
{0x010CE3, 0x010CA3},
{0x010CE4, 0x010CA4},
{0x010CE5, 0x010CA5},
{0x010CE6, 0x010CA6},
{0x010CE7, 0x010CA7},
{0x010CE8, 0x010CA8},
{0x010CE9, 0x010CA9},
{0x010CEA, 0x010CAA},
{0x010CEB, 0x010CAB},
{0x010CEC, 0x010CAC},
{0x010CED, 0x010CAD},
{0x010CEE, 0x010CAE},
{0x010CEF, 0x010CAF},
{0x010CF0, 0x010CB0},
{0x010CF1, 0x010CB1},
{0x010CF2, 0x010CB2},
{0x0118C0, 0x0118A0},
{0x0118C1, 0x0118A1},
{0x0118C2, 0x0118A2},
{0x0118C3, 0x0118A3},
{0x0118C4, 0x0118A4},
{0x0118C5, 0x0118A5},
{0x0118C6, 0x0118A6},
{0x0118C7, 0x0118A7},
{0x0118C8, 0x0118A8},
{0x0118C9, 0x0118A9},
{0x0118CA, 0x0118AA},
{0x0118CB, 0x0118AB},
{0x0118CC, 0x0118AC},
{0x0118CD, 0x0118AD},
{0x0118CE, 0x0118AE},
{0x0118CF, 0x0118AF},
{0x0118D0, 0x0118B0},
{0x0118D1, 0x0118B1},
{0x0118D2, 0x0118B2},
{0x0118D3, 0x0118B3},
{0x0118D4, 0x0118B4},
{0x0118D5, 0x0118B5},
{0x0118D6, 0x0118B6},
{0x0118D7, 0x0118B7},
{0x0118D8, 0x0118B8},
{0x0118D9, 0x0118B9},
{0x0118DA, 0x0118BA},
{0x0118DB, 0x0118BB},
{0x0118DC, 0x0118BC},
{0x0118DD, 0x0118BD},
{0x0118DE, 0x0118BE},
{0x0118DF, 0x0118BF},
{0x016E60, 0x016E40},
{0x016E61, 0x016E41},
{0x016E62, 0x016E42},
{0x016E63, 0x016E43},
{0x016E64, 0x016E44},
{0x016E65, 0x016E45},
{0x016E66, 0x016E46},
{0x016E67, 0x016E47},
{0x016E68, 0x016E48},
{0x016E69, 0x016E49},
{0x016E6A, 0x016E4A},
{0x016E6B, 0x016E4B},
{0x016E6C, 0x016E4C},
{0x016E6D, 0x016E4D},
{0x016E6E, 0x016E4E},
{0x016E6F, 0x016E4F},
{0x016E70, 0x016E50},
{0x016E71, 0x016E51},
{0x016E72, 0x016E52},
{0x016E73, 0x016E53},
{0x016E74, 0x016E54},
{0x016E75, 0x016E55},
{0x016E76, 0x016E56},
{0x016E77, 0x016E57},
{0x016E78, 0x016E58},
{0x016E79, 0x016E59},
{0x016E7A, 0x016E5A},
{0x016E7B, 0x016E5B},
{0x016E7C, 0x016E5C},
{0x016E7D, 0x016E5D},
{0x016E7E, 0x016E5E},
{0x016E7F, 0x016E5F},
{0x01E922, 0x01E900},
{0x01E923, 0x01E901},
{0x01E924, 0x01E902},
{0x01E925, 0x01E903},
{0x01E926, 0x01E904},
{0x01E927, 0x01E905},
{0x01E928, 0x01E906},
{0x01E929, 0x01E907},
{0x01E92A, 0x01E908},
{0x01E92B, 0x01E909},
{0x01E92C, 0x01E90A},
{0x01E92D, 0x01E90B},
{0x01E92E, 0x01E90C},
{0x01E92F, 0x01E90D},
{0x01E930, 0x01E90E},
{0x01E931, 0x01E90F},
{0x01E932, 0x01E910},
{0x01E933, 0x01E911},
{0x01E934, 0x01E912},
{0x01E935, 0x01E913},
{0x01E936, 0x01E914},
{0x01E937, 0x01E915},
{0x01E938, 0x01E916},
{0x01E939, 0x01E917},
{0x01E93A, 0x01E918},
{0x01E93B, 0x01E919},
{0x01E93C, 0x01E91A},
{0x01E93D, 0x01E91B},
{0x01E93E, 0x01E91C},
{0x01E93F, 0x01E91D},
{0x01E940, 0x01E91E},
{0x01E941, 0x01E91F},
{0x01E942, 0x01E920},
{0x01E943, 0x01E921},
};

const std::initializer_list<range_nfd> unicode_ranges_nfd = {  // start, last, nfd
{0x000000, 0x000000, 0x000000},
{0x0000C0, 0x0000C5, 0x000041},
{0x0000C7, 0x0000C7, 0x000043},
{0x0000C8, 0x0000CB, 0x000045},
{0x0000CC, 0x0000CF, 0x000049},
{0x0000D1, 0x0000D1, 0x00004E},
{0x0000D2, 0x0000D6, 0x00004F},
{0x0000D9, 0x0000DC, 0x000055},
{0x0000DD, 0x0000DD, 0x000059},
{0x0000E0, 0x0000E5, 0x000061},
{0x0000E7, 0x0000E7, 0x000063},
{0x0000E8, 0x0000EB, 0x000065},
{0x0000EC, 0x0000EF, 0x000069},
{0x0000F1, 0x0000F1, 0x00006E},
{0x0000F2, 0x0000F6, 0x00006F},
{0x0000F9, 0x0000FC, 0x000075},
{0x0000FD, 0x0000FD, 0x000079},
{0x0000FF, 0x0000FF, 0x000079},
{0x000100, 0x000100, 0x000041},
{0x000101, 0x000101, 0x000061},
{0x000102, 0x000102, 0x000041},
{0x000103, 0x000103, 0x000061},
{0x000104, 0x000104, 0x000041},
{0x000105, 0x000105, 0x000061},
{0x000106, 0x000106, 0x000043},
{0x000107, 0x000107, 0x000063},
{0x000108, 0x000108, 0x000043},
{0x000109, 0x000109, 0x000063},
{0x00010A, 0x00010A, 0x000043},
{0x00010B, 0x00010B, 0x000063},
{0x00010C, 0x00010C, 0x000043},
{0x00010D, 0x00010D, 0x000063},
{0x00010E, 0x00010E, 0x000044},
{0x00010F, 0x00010F, 0x000064},
{0x000112, 0x000112, 0x000045},
{0x000113, 0x000113, 0x000065},
{0x000114, 0x000114, 0x000045},
{0x000115, 0x000115, 0x000065},
{0x000116, 0x000116, 0x000045},
{0x000117, 0x000117, 0x000065},
{0x000118, 0x000118, 0x000045},
{0x000119, 0x000119, 0x000065},
{0x00011A, 0x00011A, 0x000045},
{0x00011B, 0x00011B, 0x000065},
{0x00011C, 0x00011C, 0x000047},
{0x00011D, 0x00011D, 0x000067},
{0x00011E, 0x00011E, 0x000047},
{0x00011F, 0x00011F, 0x000067},
{0x000120, 0x000120, 0x000047},
{0x000121, 0x000121, 0x000067},
{0x000122, 0x000122, 0x000047},
{0x000123, 0x000123, 0x000067},
{0x000124, 0x000124, 0x000048},
{0x000125, 0x000125, 0x000068},
{0x000128, 0x000128, 0x000049},
{0x000129, 0x000129, 0x000069},
{0x00012A, 0x00012A, 0x000049},
{0x00012B, 0x00012B, 0x000069},
{0x00012C, 0x00012C, 0x000049},
{0x00012D, 0x00012D, 0x000069},
{0x00012E, 0x00012E, 0x000049},
{0x00012F, 0x00012F, 0x000069},
{0x000130, 0x000130, 0x000049},
{0x000134, 0x000134, 0x00004A},
{0x000135, 0x000135, 0x00006A},
{0x000136, 0x000136, 0x00004B},
{0x000137, 0x000137, 0x00006B},
{0x000139, 0x000139, 0x00004C},
{0x00013A, 0x00013A, 0x00006C},
{0x00013B, 0x00013B, 0x00004C},
{0x00013C, 0x00013C, 0x00006C},
{0x00013D, 0x00013D, 0x00004C},
{0x00013E, 0x00013E, 0x00006C},
{0x000143, 0x000143, 0x00004E},
{0x000144, 0x000144, 0x00006E},
{0x000145, 0x000145, 0x00004E},
{0x000146, 0x000146, 0x00006E},
{0x000147, 0x000147, 0x00004E},
{0x000148, 0x000148, 0x00006E},
{0x00014C, 0x00014C, 0x00004F},
{0x00014D, 0x00014D, 0x00006F},
{0x00014E, 0x00014E, 0x00004F},
{0x00014F, 0x00014F, 0x00006F},
{0x000150, 0x000150, 0x00004F},
{0x000151, 0x000151, 0x00006F},
{0x000154, 0x000154, 0x000052},
{0x000155, 0x000155, 0x000072},
{0x000156, 0x000156, 0x000052},
{0x000157, 0x000157, 0x000072},
{0x000158, 0x000158, 0x000052},
{0x000159, 0x000159, 0x000072},
{0x00015A, 0x00015A, 0x000053},
{0x00015B, 0x00015B, 0x000073},
{0x00015C, 0x00015C, 0x000053},
{0x00015D, 0x00015D, 0x000073},
{0x00015E, 0x00015E, 0x000053},
{0x00015F, 0x00015F, 0x000073},
{0x000160, 0x000160, 0x000053},
{0x000161, 0x000161, 0x000073},
{0x000162, 0x000162, 0x000054},
{0x000163, 0x000163, 0x000074},
{0x000164, 0x000164, 0x000054},
{0x000165, 0x000165, 0x000074},
{0x000168, 0x000168, 0x000055},
{0x000169, 0x000169, 0x000075},
{0x00016A, 0x00016A, 0x000055},
{0x00016B, 0x00016B, 0x000075},
{0x00016C, 0x00016C, 0x000055},
{0x00016D, 0x00016D, 0x000075},
{0x00016E, 0x00016E, 0x000055},
{0x00016F, 0x00016F, 0x000075},
{0x000170, 0x000170, 0x000055},
{0x000171, 0x000171, 0x000075},
{0x000172, 0x000172, 0x000055},
{0x000173, 0x000173, 0x000075},
{0x000174, 0x000174, 0x000057},
{0x000175, 0x000175, 0x000077},
{0x000176, 0x000176, 0x000059},
{0x000177, 0x000177, 0x000079},
{0x000178, 0x000178, 0x000059},
{0x000179, 0x000179, 0x00005A},
{0x00017A, 0x00017A, 0x00007A},
{0x00017B, 0x00017B, 0x00005A},
{0x00017C, 0x00017C, 0x00007A},
{0x00017D, 0x00017D, 0x00005A},
{0x00017E, 0x00017E, 0x00007A},
{0x0001A0, 0x0001A0, 0x00004F},
{0x0001A1, 0x0001A1, 0x00006F},
{0x0001AF, 0x0001AF, 0x000055},
{0x0001B0, 0x0001B0, 0x000075},
{0x0001CD, 0x0001CD, 0x000041},
{0x0001CE, 0x0001CE, 0x000061},
{0x0001CF, 0x0001CF, 0x000049},
{0x0001D0, 0x0001D0, 0x000069},
{0x0001D1, 0x0001D1, 0x00004F},
{0x0001D2, 0x0001D2, 0x00006F},
{0x0001D3, 0x0001D3, 0x000055},
{0x0001D4, 0x0001D4, 0x000075},
{0x0001D5, 0x0001D5, 0x000055},
{0x0001D6, 0x0001D6, 0x000075},
{0x0001D7, 0x0001D7, 0x000055},
{0x0001D8, 0x0001D8, 0x000075},
{0x0001D9, 0x0001D9, 0x000055},
{0x0001DA, 0x0001DA, 0x000075},
{0x0001DB, 0x0001DB, 0x000055},
{0x0001DC, 0x0001DC, 0x000075},
{0x0001DE, 0x0001DE, 0x000041},
{0x0001DF, 0x0001DF, 0x000061},
{0x0001E0, 0x0001E0, 0x000041},
{0x0001E1, 0x0001E1, 0x000061},
{0x0001E2, 0x0001E2, 0x0000C6},
{0x0001E3, 0x0001E3, 0x0000E6},
{0x0001E6, 0x0001E6, 0x000047},
{0x0001E7, 0x0001E7, 0x000067},
{0x0001E8, 0x0001E8, 0x00004B},
{0x0001E9, 0x0001E9, 0x00006B},
{0x0001EA, 0x0001EA, 0x00004F},
{0x0001EB, 0x0001EB, 0x00006F},
{0x0001EC, 0x0001EC, 0x00004F},
{0x0001ED, 0x0001ED, 0x00006F},
{0x0001EE, 0x0001EE, 0x0001B7},
{0x0001EF, 0x0001EF, 0x000292},
{0x0001F0, 0x0001F0, 0x00006A},
{0x0001F4, 0x0001F4, 0x000047},
{0x0001F5, 0x0001F5, 0x000067},
{0x0001F8, 0x0001F8, 0x00004E},
{0x0001F9, 0x0001F9, 0x00006E},
{0x0001FA, 0x0001FA, 0x000041},
{0x0001FB, 0x0001FB, 0x000061},
{0x0001FC, 0x0001FC, 0x0000C6},
{0x0001FD, 0x0001FD, 0x0000E6},
{0x0001FE, 0x0001FE, 0x0000D8},
{0x0001FF, 0x0001FF, 0x0000F8},
{0x000200, 0x000200, 0x000041},
{0x000201, 0x000201, 0x000061},
{0x000202, 0x000202, 0x000041},
{0x000203, 0x000203, 0x000061},
{0x000204, 0x000204, 0x000045},
{0x000205, 0x000205, 0x000065},
{0x000206, 0x000206, 0x000045},
{0x000207, 0x000207, 0x000065},
{0x000208, 0x000208, 0x000049},
{0x000209, 0x000209, 0x000069},
{0x00020A, 0x00020A, 0x000049},
{0x00020B, 0x00020B, 0x000069},
{0x00020C, 0x00020C, 0x00004F},
{0x00020D, 0x00020D, 0x00006F},
{0x00020E, 0x00020E, 0x00004F},
{0x00020F, 0x00020F, 0x00006F},
{0x000210, 0x000210, 0x000052},
{0x000211, 0x000211, 0x000072},
{0x000212, 0x000212, 0x000052},
{0x000213, 0x000213, 0x000072},
{0x000214, 0x000214, 0x000055},
{0x000215, 0x000215, 0x000075},
{0x000216, 0x000216, 0x000055},
{0x000217, 0x000217, 0x000075},
{0x000218, 0x000218, 0x000053},
{0x000219, 0x000219, 0x000073},
{0x00021A, 0x00021A, 0x000054},
{0x00021B, 0x00021B, 0x000074},
{0x00021E, 0x00021E, 0x000048},
{0x00021F, 0x00021F, 0x000068},
{0x000226, 0x000226, 0x000041},
{0x000227, 0x000227, 0x000061},
{0x000228, 0x000228, 0x000045},
{0x000229, 0x000229, 0x000065},
{0x00022A, 0x00022A, 0x00004F},
{0x00022B, 0x00022B, 0x00006F},
{0x00022C, 0x00022C, 0x00004F},
{0x00022D, 0x00022D, 0x00006F},
{0x00022E, 0x00022E, 0x00004F},
{0x00022F, 0x00022F, 0x00006F},
{0x000230, 0x000230, 0x00004F},
{0x000231, 0x000231, 0x00006F},
{0x000232, 0x000232, 0x000059},
{0x000233, 0x000233, 0x000079},
{0x000340, 0x000340, 0x000300},
{0x000341, 0x000341, 0x000301},
{0x000343, 0x000343, 0x000313},
{0x000344, 0x000344, 0x000308},
{0x000374, 0x000374, 0x0002B9},
{0x00037E, 0x00037E, 0x00003B},
{0x000385, 0x000385, 0x0000A8},
{0x000386, 0x000386, 0x000391},
{0x000387, 0x000387, 0x0000B7},
{0x000388, 0x000388, 0x000395},
{0x000389, 0x000389, 0x000397},
{0x00038A, 0x00038A, 0x000399},
{0x00038C, 0x00038C, 0x00039F},
{0x00038E, 0x00038E, 0x0003A5},
{0x00038F, 0x00038F, 0x0003A9},
{0x000390, 0x000390, 0x0003B9},
{0x0003AA, 0x0003AA, 0x000399},
{0x0003AB, 0x0003AB, 0x0003A5},
{0x0003AC, 0x0003AC, 0x0003B1},
{0x0003AD, 0x0003AD, 0x0003B5},
{0x0003AE, 0x0003AE, 0x0003B7},
{0x0003AF, 0x0003AF, 0x0003B9},
{0x0003B0, 0x0003B0, 0x0003C5},
{0x0003CA, 0x0003CA, 0x0003B9},
{0x0003CB, 0x0003CB, 0x0003C5},
{0x0003CC, 0x0003CC, 0x0003BF},
{0x0003CD, 0x0003CD, 0x0003C5},
{0x0003CE, 0x0003CE, 0x0003C9},
{0x0003D3, 0x0003D4, 0x0003D2},
{0x000400, 0x000401, 0x000415},
{0x000403, 0x000403, 0x000413},
{0x000407, 0x000407, 0x000406},
{0x00040C, 0x00040C, 0x00041A},
{0x00040D, 0x00040D, 0x000418},
{0x00040E, 0x00040E, 0x000423},
{0x000419, 0x000419, 0x000418},
{0x000439, 0x000439, 0x000438},
{0x000450, 0x000451, 0x000435},
{0x000453, 0x000453, 0x000433},
{0x000457, 0x000457, 0x000456},
{0x00045C, 0x00045C, 0x00043A},
{0x00045D, 0x00045D, 0x000438},
{0x00045E, 0x00045E, 0x000443},
{0x000476, 0x000476, 0x000474},
{0x000477, 0x000477, 0x000475},
{0x0004C1, 0x0004C1, 0x000416},
{0x0004C2, 0x0004C2, 0x000436},
{0x0004D0, 0x0004D0, 0x000410},
{0x0004D1, 0x0004D1, 0x000430},
{0x0004D2, 0x0004D2, 0x000410},
{0x0004D3, 0x0004D3, 0x000430},
{0x0004D6, 0x0004D6, 0x000415},
{0x0004D7, 0x0004D7, 0x000435},
{0x0004DA, 0x0004DA, 0x0004D8},
{0x0004DB, 0x0004DB, 0x0004D9},
{0x0004DC, 0x0004DC, 0x000416},
{0x0004DD, 0x0004DD, 0x000436},
{0x0004DE, 0x0004DE, 0x000417},
{0x0004DF, 0x0004DF, 0x000437},
{0x0004E2, 0x0004E2, 0x000418},
{0x0004E3, 0x0004E3, 0x000438},
{0x0004E4, 0x0004E4, 0x000418},
{0x0004E5, 0x0004E5, 0x000438},
{0x0004E6, 0x0004E6, 0x00041E},
{0x0004E7, 0x0004E7, 0x00043E},
{0x0004EA, 0x0004EA, 0x0004E8},
{0x0004EB, 0x0004EB, 0x0004E9},
{0x0004EC, 0x0004EC, 0x00042D},
{0x0004ED, 0x0004ED, 0x00044D},
{0x0004EE, 0x0004EE, 0x000423},
{0x0004EF, 0x0004EF, 0x000443},
{0x0004F0, 0x0004F0, 0x000423},
{0x0004F1, 0x0004F1, 0x000443},
{0x0004F2, 0x0004F2, 0x000423},
{0x0004F3, 0x0004F3, 0x000443},
{0x0004F4, 0x0004F4, 0x000427},
{0x0004F5, 0x0004F5, 0x000447},
{0x0004F8, 0x0004F8, 0x00042B},
{0x0004F9, 0x0004F9, 0x00044B},
{0x000622, 0x000623, 0x000627},
{0x000624, 0x000624, 0x000648},
{0x000625, 0x000625, 0x000627},
{0x000626, 0x000626, 0x00064A},
{0x0006C0, 0x0006C0, 0x0006D5},
{0x0006C2, 0x0006C2, 0x0006C1},
{0x0006D3, 0x0006D3, 0x0006D2},
{0x000929, 0x000929, 0x000928},
{0x000931, 0x000931, 0x000930},
{0x000934, 0x000934, 0x000933},
{0x000958, 0x000958, 0x000915},
{0x000959, 0x000959, 0x000916},
{0x00095A, 0x00095A, 0x000917},
{0x00095B, 0x00095B, 0x00091C},
{0x00095C, 0x00095C, 0x000921},
{0x00095D, 0x00095D, 0x000922},
{0x00095E, 0x00095E, 0x00092B},
{0x00095F, 0x00095F, 0x00092F},
{0x0009CB, 0x0009CC, 0x0009C7},
{0x0009DC, 0x0009DC, 0x0009A1},
{0x0009DD, 0x0009DD, 0x0009A2},
{0x0009DF, 0x0009DF, 0x0009AF},
{0x000A33, 0x000A33, 0x000A32},
{0x000A36, 0x000A36, 0x000A38},
{0x000A59, 0x000A59, 0x000A16},
{0x000A5A, 0x000A5A, 0x000A17},
{0x000A5B, 0x000A5B, 0x000A1C},
{0x000A5E, 0x000A5E, 0x000A2B},
{0x000B48, 0x000B48, 0x000B47},
{0x000B4B, 0x000B4C, 0x000B47},
{0x000B5C, 0x000B5C, 0x000B21},
{0x000B5D, 0x000B5D, 0x000B22},
{0x000B94, 0x000B94, 0x000B92},
{0x000BCA, 0x000BCA, 0x000BC6},
{0x000BCB, 0x000BCB, 0x000BC7},
{0x000BCC, 0x000BCC, 0x000BC6},
{0x000C48, 0x000C48, 0x000C46},
{0x000CC0, 0x000CC0, 0x000CBF},
{0x000CC7, 0x000CC8, 0x000CC6},
{0x000CCA, 0x000CCB, 0x000CC6},
{0x000D4A, 0x000D4A, 0x000D46},
{0x000D4B, 0x000D4B, 0x000D47},
{0x000D4C, 0x000D4C, 0x000D46},
{0x000DDA, 0x000DDA, 0x000DD9},
{0x000DDC, 0x000DDE, 0x000DD9},
{0x000F43, 0x000F43, 0x000F42},
{0x000F4D, 0x000F4D, 0x000F4C},
{0x000F52, 0x000F52, 0x000F51},
{0x000F57, 0x000F57, 0x000F56},
{0x000F5C, 0x000F5C, 0x000F5B},
{0x000F69, 0x000F69, 0x000F40},
{0x000F73, 0x000F73, 0x000F71},
{0x000F75, 0x000F75, 0x000F71},
{0x000F76, 0x000F76, 0x000FB2},
{0x000F78, 0x000F78, 0x000FB3},
{0x000F81, 0x000F81, 0x000F71},
{0x000F93, 0x000F93, 0x000F92},
{0x000F9D, 0x000F9D, 0x000F9C},
{0x000FA2, 0x000FA2, 0x000FA1},
{0x000FA7, 0x000FA7, 0x000FA6},
{0x000FAC, 0x000FAC, 0x000FAB},
{0x000FB9, 0x000FB9, 0x000F90},
{0x001026, 0x001026, 0x001025},
{0x001B06, 0x001B06, 0x001B05},
{0x001B08, 0x001B08, 0x001B07},
{0x001B0A, 0x001B0A, 0x001B09},
{0x001B0C, 0x001B0C, 0x001B0B},
{0x001B0E, 0x001B0E, 0x001B0D},
{0x001B12, 0x001B12, 0x001B11},
{0x001B3B, 0x001B3B, 0x001B3A},
{0x001B3D, 0x001B3D, 0x001B3C},
{0x001B40, 0x001B40, 0x001B3E},
{0x001B41, 0x001B41, 0x001B3F},
{0x001B43, 0x001B43, 0x001B42},
{0x001E00, 0x001E00, 0x000041},
{0x001E01, 0x001E01, 0x000061},
{0x001E02, 0x001E02, 0x000042},
{0x001E03, 0x001E03, 0x000062},
{0x001E04, 0x001E04, 0x000042},
{0x001E05, 0x001E05, 0x000062},
{0x001E06, 0x001E06, 0x000042},
{0x001E07, 0x001E07, 0x000062},
{0x001E08, 0x001E08, 0x000043},
{0x001E09, 0x001E09, 0x000063},
{0x001E0A, 0x001E0A, 0x000044},
{0x001E0B, 0x001E0B, 0x000064},
{0x001E0C, 0x001E0C, 0x000044},
{0x001E0D, 0x001E0D, 0x000064},
{0x001E0E, 0x001E0E, 0x000044},
{0x001E0F, 0x001E0F, 0x000064},
{0x001E10, 0x001E10, 0x000044},
{0x001E11, 0x001E11, 0x000064},
{0x001E12, 0x001E12, 0x000044},
{0x001E13, 0x001E13, 0x000064},
{0x001E14, 0x001E14, 0x000045},
{0x001E15, 0x001E15, 0x000065},
{0x001E16, 0x001E16, 0x000045},
{0x001E17, 0x001E17, 0x000065},
{0x001E18, 0x001E18, 0x000045},
{0x001E19, 0x001E19, 0x000065},
{0x001E1A, 0x001E1A, 0x000045},
{0x001E1B, 0x001E1B, 0x000065},
{0x001E1C, 0x001E1C, 0x000045},
{0x001E1D, 0x001E1D, 0x000065},
{0x001E1E, 0x001E1E, 0x000046},
{0x001E1F, 0x001E1F, 0x000066},
{0x001E20, 0x001E20, 0x000047},
{0x001E21, 0x001E21, 0x000067},
{0x001E22, 0x001E22, 0x000048},
{0x001E23, 0x001E23, 0x000068},
{0x001E24, 0x001E24, 0x000048},
{0x001E25, 0x001E25, 0x000068},
{0x001E26, 0x001E26, 0x000048},
{0x001E27, 0x001E27, 0x000068},
{0x001E28, 0x001E28, 0x000048},
{0x001E29, 0x001E29, 0x000068},
{0x001E2A, 0x001E2A, 0x000048},
{0x001E2B, 0x001E2B, 0x000068},
{0x001E2C, 0x001E2C, 0x000049},
{0x001E2D, 0x001E2D, 0x000069},
{0x001E2E, 0x001E2E, 0x000049},
{0x001E2F, 0x001E2F, 0x000069},
{0x001E30, 0x001E30, 0x00004B},
{0x001E31, 0x001E31, 0x00006B},
{0x001E32, 0x001E32, 0x00004B},
{0x001E33, 0x001E33, 0x00006B},
{0x001E34, 0x001E34, 0x00004B},
{0x001E35, 0x001E35, 0x00006B},
{0x001E36, 0x001E36, 0x00004C},
{0x001E37, 0x001E37, 0x00006C},
{0x001E38, 0x001E38, 0x00004C},
{0x001E39, 0x001E39, 0x00006C},
{0x001E3A, 0x001E3A, 0x00004C},
{0x001E3B, 0x001E3B, 0x00006C},
{0x001E3C, 0x001E3C, 0x00004C},
{0x001E3D, 0x001E3D, 0x00006C},
{0x001E3E, 0x001E3E, 0x00004D},
{0x001E3F, 0x001E3F, 0x00006D},
{0x001E40, 0x001E40, 0x00004D},
{0x001E41, 0x001E41, 0x00006D},
{0x001E42, 0x001E42, 0x00004D},
{0x001E43, 0x001E43, 0x00006D},
{0x001E44, 0x001E44, 0x00004E},
{0x001E45, 0x001E45, 0x00006E},
{0x001E46, 0x001E46, 0x00004E},
{0x001E47, 0x001E47, 0x00006E},
{0x001E48, 0x001E48, 0x00004E},
{0x001E49, 0x001E49, 0x00006E},
{0x001E4A, 0x001E4A, 0x00004E},
{0x001E4B, 0x001E4B, 0x00006E},
{0x001E4C, 0x001E4C, 0x00004F},
{0x001E4D, 0x001E4D, 0x00006F},
{0x001E4E, 0x001E4E, 0x00004F},
{0x001E4F, 0x001E4F, 0x00006F},
{0x001E50, 0x001E50, 0x00004F},
{0x001E51, 0x001E51, 0x00006F},
{0x001E52, 0x001E52, 0x00004F},
{0x001E53, 0x001E53, 0x00006F},
{0x001E54, 0x001E54, 0x000050},
{0x001E55, 0x001E55, 0x000070},
{0x001E56, 0x001E56, 0x000050},
{0x001E57, 0x001E57, 0x000070},
{0x001E58, 0x001E58, 0x000052},
{0x001E59, 0x001E59, 0x000072},
{0x001E5A, 0x001E5A, 0x000052},
{0x001E5B, 0x001E5B, 0x000072},
{0x001E5C, 0x001E5C, 0x000052},
{0x001E5D, 0x001E5D, 0x000072},
{0x001E5E, 0x001E5E, 0x000052},
{0x001E5F, 0x001E5F, 0x000072},
{0x001E60, 0x001E60, 0x000053},
{0x001E61, 0x001E61, 0x000073},
{0x001E62, 0x001E62, 0x000053},
{0x001E63, 0x001E63, 0x000073},
{0x001E64, 0x001E64, 0x000053},
{0x001E65, 0x001E65, 0x000073},
{0x001E66, 0x001E66, 0x000053},
{0x001E67, 0x001E67, 0x000073},
{0x001E68, 0x001E68, 0x000053},
{0x001E69, 0x001E69, 0x000073},
{0x001E6A, 0x001E6A, 0x000054},
{0x001E6B, 0x001E6B, 0x000074},
{0x001E6C, 0x001E6C, 0x000054},
{0x001E6D, 0x001E6D, 0x000074},
{0x001E6E, 0x001E6E, 0x000054},
{0x001E6F, 0x001E6F, 0x000074},
{0x001E70, 0x001E70, 0x000054},
{0x001E71, 0x001E71, 0x000074},
{0x001E72, 0x001E72, 0x000055},
{0x001E73, 0x001E73, 0x000075},
{0x001E74, 0x001E74, 0x000055},
{0x001E75, 0x001E75, 0x000075},
{0x001E76, 0x001E76, 0x000055},
{0x001E77, 0x001E77, 0x000075},
{0x001E78, 0x001E78, 0x000055},
{0x001E79, 0x001E79, 0x000075},
{0x001E7A, 0x001E7A, 0x000055},
{0x001E7B, 0x001E7B, 0x000075},
{0x001E7C, 0x001E7C, 0x000056},
{0x001E7D, 0x001E7D, 0x000076},
{0x001E7E, 0x001E7E, 0x000056},
{0x001E7F, 0x001E7F, 0x000076},
{0x001E80, 0x001E80, 0x000057},
{0x001E81, 0x001E81, 0x000077},
{0x001E82, 0x001E82, 0x000057},
{0x001E83, 0x001E83, 0x000077},
{0x001E84, 0x001E84, 0x000057},
{0x001E85, 0x001E85, 0x000077},
{0x001E86, 0x001E86, 0x000057},
{0x001E87, 0x001E87, 0x000077},
{0x001E88, 0x001E88, 0x000057},
{0x001E89, 0x001E89, 0x000077},
{0x001E8A, 0x001E8A, 0x000058},
{0x001E8B, 0x001E8B, 0x000078},
{0x001E8C, 0x001E8C, 0x000058},
{0x001E8D, 0x001E8D, 0x000078},
{0x001E8E, 0x001E8E, 0x000059},
{0x001E8F, 0x001E8F, 0x000079},
{0x001E90, 0x001E90, 0x00005A},
{0x001E91, 0x001E91, 0x00007A},
{0x001E92, 0x001E92, 0x00005A},
{0x001E93, 0x001E93, 0x00007A},
{0x001E94, 0x001E94, 0x00005A},
{0x001E95, 0x001E95, 0x00007A},
{0x001E96, 0x001E96, 0x000068},
{0x001E97, 0x001E97, 0x000074},
{0x001E98, 0x001E98, 0x000077},
{0x001E99, 0x001E99, 0x000079},
{0x001E9B, 0x001E9B, 0x00017F},
{0x001EA0, 0x001EA0, 0x000041},
{0x001EA1, 0x001EA1, 0x000061},
{0x001EA2, 0x001EA2, 0x000041},
{0x001EA3, 0x001EA3, 0x000061},
{0x001EA4, 0x001EA4, 0x000041},
{0x001EA5, 0x001EA5, 0x000061},
{0x001EA6, 0x001EA6, 0x000041},
{0x001EA7, 0x001EA7, 0x000061},
{0x001EA8, 0x001EA8, 0x000041},
{0x001EA9, 0x001EA9, 0x000061},
{0x001EAA, 0x001EAA, 0x000041},
{0x001EAB, 0x001EAB, 0x000061},
{0x001EAC, 0x001EAC, 0x000041},
{0x001EAD, 0x001EAD, 0x000061},
{0x001EAE, 0x001EAE, 0x000041},
{0x001EAF, 0x001EAF, 0x000061},
{0x001EB0, 0x001EB0, 0x000041},
{0x001EB1, 0x001EB1, 0x000061},
{0x001EB2, 0x001EB2, 0x000041},
{0x001EB3, 0x001EB3, 0x000061},
{0x001EB4, 0x001EB4, 0x000041},
{0x001EB5, 0x001EB5, 0x000061},
{0x001EB6, 0x001EB6, 0x000041},
{0x001EB7, 0x001EB7, 0x000061},
{0x001EB8, 0x001EB8, 0x000045},
{0x001EB9, 0x001EB9, 0x000065},
{0x001EBA, 0x001EBA, 0x000045},
{0x001EBB, 0x001EBB, 0x000065},
{0x001EBC, 0x001EBC, 0x000045},
{0x001EBD, 0x001EBD, 0x000065},
{0x001EBE, 0x001EBE, 0x000045},
{0x001EBF, 0x001EBF, 0x000065},
{0x001EC0, 0x001EC0, 0x000045},
{0x001EC1, 0x001EC1, 0x000065},
{0x001EC2, 0x001EC2, 0x000045},
{0x001EC3, 0x001EC3, 0x000065},
{0x001EC4, 0x001EC4, 0x000045},
{0x001EC5, 0x001EC5, 0x000065},
{0x001EC6, 0x001EC6, 0x000045},
{0x001EC7, 0x001EC7, 0x000065},
{0x001EC8, 0x001EC8, 0x000049},
{0x001EC9, 0x001EC9, 0x000069},
{0x001ECA, 0x001ECA, 0x000049},
{0x001ECB, 0x001ECB, 0x000069},
{0x001ECC, 0x001ECC, 0x00004F},
{0x001ECD, 0x001ECD, 0x00006F},
{0x001ECE, 0x001ECE, 0x00004F},
{0x001ECF, 0x001ECF, 0x00006F},
{0x001ED0, 0x001ED0, 0x00004F},
{0x001ED1, 0x001ED1, 0x00006F},
{0x001ED2, 0x001ED2, 0x00004F},
{0x001ED3, 0x001ED3, 0x00006F},
{0x001ED4, 0x001ED4, 0x00004F},
{0x001ED5, 0x001ED5, 0x00006F},
{0x001ED6, 0x001ED6, 0x00004F},
{0x001ED7, 0x001ED7, 0x00006F},
{0x001ED8, 0x001ED8, 0x00004F},
{0x001ED9, 0x001ED9, 0x00006F},
{0x001EDA, 0x001EDA, 0x00004F},
{0x001EDB, 0x001EDB, 0x00006F},
{0x001EDC, 0x001EDC, 0x00004F},
{0x001EDD, 0x001EDD, 0x00006F},
{0x001EDE, 0x001EDE, 0x00004F},
{0x001EDF, 0x001EDF, 0x00006F},
{0x001EE0, 0x001EE0, 0x00004F},
{0x001EE1, 0x001EE1, 0x00006F},
{0x001EE2, 0x001EE2, 0x00004F},
{0x001EE3, 0x001EE3, 0x00006F},
{0x001EE4, 0x001EE4, 0x000055},
{0x001EE5, 0x001EE5, 0x000075},
{0x001EE6, 0x001EE6, 0x000055},
{0x001EE7, 0x001EE7, 0x000075},
{0x001EE8, 0x001EE8, 0x000055},
{0x001EE9, 0x001EE9, 0x000075},
{0x001EEA, 0x001EEA, 0x000055},
{0x001EEB, 0x001EEB, 0x000075},
{0x001EEC, 0x001EEC, 0x000055},
{0x001EED, 0x001EED, 0x000075},
{0x001EEE, 0x001EEE, 0x000055},
{0x001EEF, 0x001EEF, 0x000075},
{0x001EF0, 0x001EF0, 0x000055},
{0x001EF1, 0x001EF1, 0x000075},
{0x001EF2, 0x001EF2, 0x000059},
{0x001EF3, 0x001EF3, 0x000079},
{0x001EF4, 0x001EF4, 0x000059},
{0x001EF5, 0x001EF5, 0x000079},
{0x001EF6, 0x001EF6, 0x000059},
{0x001EF7, 0x001EF7, 0x000079},
{0x001EF8, 0x001EF8, 0x000059},
{0x001EF9, 0x001EF9, 0x000079},
{0x001F00, 0x001F07, 0x0003B1},
{0x001F08, 0x001F0F, 0x000391},
{0x001F10, 0x001F15, 0x0003B5},
{0x001F18, 0x001F1D, 0x000395},
{0x001F20, 0x001F27, 0x0003B7},
{0x001F28, 0x001F2F, 0x000397},
{0x001F30, 0x001F37, 0x0003B9},
{0x001F38, 0x001F3F, 0x000399},
{0x001F40, 0x001F45, 0x0003BF},
{0x001F48, 0x001F4D, 0x00039F},
{0x001F50, 0x001F57, 0x0003C5},
{0x001F59, 0x001F59, 0x0003A5},
{0x001F5B, 0x001F5B, 0x0003A5},
{0x001F5D, 0x001F5D, 0x0003A5},
{0x001F5F, 0x001F5F, 0x0003A5},
{0x001F60, 0x001F67, 0x0003C9},
{0x001F68, 0x001F6F, 0x0003A9},
{0x001F70, 0x001F71, 0x0003B1},
{0x001F72, 0x001F73, 0x0003B5},
{0x001F74, 0x001F75, 0x0003B7},
{0x001F76, 0x001F77, 0x0003B9},
{0x001F78, 0x001F79, 0x0003BF},
{0x001F7A, 0x001F7B, 0x0003C5},
{0x001F7C, 0x001F7D, 0x0003C9},
{0x001F80, 0x001F87, 0x0003B1},
{0x001F88, 0x001F8F, 0x000391},
{0x001F90, 0x001F97, 0x0003B7},
{0x001F98, 0x001F9F, 0x000397},
{0x001FA0, 0x001FA7, 0x0003C9},
{0x001FA8, 0x001FAF, 0x0003A9},
{0x001FB0, 0x001FB4, 0x0003B1},
{0x001FB6, 0x001FB7, 0x0003B1},
{0x001FB8, 0x001FBC, 0x000391},
{0x001FBE, 0x001FBE, 0x0003B9},
{0x001FC1, 0x001FC1, 0x0000A8},
{0x001FC2, 0x001FC4, 0x0003B7},
{0x001FC6, 0x001FC7, 0x0003B7},
{0x001FC8, 0x001FC9, 0x000395},
{0x001FCA, 0x001FCC, 0x000397},
{0x001FCD, 0x001FCF, 0x001FBF},
{0x001FD0, 0x001FD3, 0x0003B9},
{0x001FD6, 0x001FD7, 0x0003B9},
{0x001FD8, 0x001FDB, 0x000399},
{0x001FDD, 0x001FDF, 0x001FFE},
{0x001FE0, 0x001FE3, 0x0003C5},
{0x001FE4, 0x001FE5, 0x0003C1},
{0x001FE6, 0x001FE7, 0x0003C5},
{0x001FE8, 0x001FEB, 0x0003A5},
{0x001FEC, 0x001FEC, 0x0003A1},
{0x001FED, 0x001FEE, 0x0000A8},
{0x001FEF, 0x001FEF, 0x000060},
{0x001FF2, 0x001FF4, 0x0003C9},
{0x001FF6, 0x001FF7, 0x0003C9},
{0x001FF8, 0x001FF9, 0x00039F},
{0x001FFA, 0x001FFC, 0x0003A9},
{0x001FFD, 0x001FFD, 0x0000B4},
{0x002000, 0x002000, 0x002002},
{0x002001, 0x002001, 0x002003},
{0x002126, 0x002126, 0x0003A9},
{0x00212A, 0x00212A, 0x00004B},
{0x00212B, 0x00212B, 0x000041},
{0x00219A, 0x00219A, 0x002190},
{0x00219B, 0x00219B, 0x002192},
{0x0021AE, 0x0021AE, 0x002194},
{0x0021CD, 0x0021CD, 0x0021D0},
{0x0021CE, 0x0021CE, 0x0021D4},
{0x0021CF, 0x0021CF, 0x0021D2},
{0x002204, 0x002204, 0x002203},
{0x002209, 0x002209, 0x002208},
{0x00220C, 0x00220C, 0x00220B},
{0x002224, 0x002224, 0x002223},
{0x002226, 0x002226, 0x002225},
{0x002241, 0x002241, 0x00223C},
{0x002244, 0x002244, 0x002243},
{0x002247, 0x002247, 0x002245},
{0x002249, 0x002249, 0x002248},
{0x002260, 0x002260, 0x00003D},
{0x002262, 0x002262, 0x002261},
{0x00226D, 0x00226D, 0x00224D},
{0x00226E, 0x00226E, 0x00003C},
{0x00226F, 0x00226F, 0x00003E},
{0x002270, 0x002270, 0x002264},
{0x002271, 0x002271, 0x002265},
{0x002274, 0x002274, 0x002272},
{0x002275, 0x002275, 0x002273},
{0x002278, 0x002278, 0x002276},
{0x002279, 0x002279, 0x002277},
{0x002280, 0x002280, 0x00227A},
{0x002281, 0x002281, 0x00227B},
{0x002284, 0x002284, 0x002282},
{0x002285, 0x002285, 0x002283},
{0x002288, 0x002288, 0x002286},
{0x002289, 0x002289, 0x002287},
{0x0022AC, 0x0022AC, 0x0022A2},
{0x0022AD, 0x0022AD, 0x0022A8},
{0x0022AE, 0x0022AE, 0x0022A9},
{0x0022AF, 0x0022AF, 0x0022AB},
{0x0022E0, 0x0022E0, 0x00227C},
{0x0022E1, 0x0022E1, 0x00227D},
{0x0022E2, 0x0022E2, 0x002291},
{0x0022E3, 0x0022E3, 0x002292},
{0x0022EA, 0x0022EA, 0x0022B2},
{0x0022EB, 0x0022EB, 0x0022B3},
{0x0022EC, 0x0022EC, 0x0022B4},
{0x0022ED, 0x0022ED, 0x0022B5},
{0x002329, 0x002329, 0x003008},
{0x00232A, 0x00232A, 0x003009},
{0x002ADC, 0x002ADC, 0x002ADD},
{0x00304C, 0x00304C, 0x00304B},
{0x00304E, 0x00304E, 0x00304D},
{0x003050, 0x003050, 0x00304F},
{0x003052, 0x003052, 0x003051},
{0x003054, 0x003054, 0x003053},
{0x003056, 0x003056, 0x003055},
{0x003058, 0x003058, 0x003057},
{0x00305A, 0x00305A, 0x003059},
{0x00305C, 0x00305C, 0x00305B},
{0x00305E, 0x00305E, 0x00305D},
{0x003060, 0x003060, 0x00305F},
{0x003062, 0x003062, 0x003061},
{0x003065, 0x003065, 0x003064},
{0x003067, 0x003067, 0x003066},
{0x003069, 0x003069, 0x003068},
{0x003070, 0x003071, 0x00306F},
{0x003073, 0x003074, 0x003072},
{0x003076, 0x003077, 0x003075},
{0x003079, 0x00307A, 0x003078},
{0x00307C, 0x00307D, 0x00307B},
{0x003094, 0x003094, 0x003046},
{0x00309E, 0x00309E, 0x00309D},
{0x0030AC, 0x0030AC, 0x0030AB},
{0x0030AE, 0x0030AE, 0x0030AD},
{0x0030B0, 0x0030B0, 0x0030AF},
{0x0030B2, 0x0030B2, 0x0030B1},
{0x0030B4, 0x0030B4, 0x0030B3},
{0x0030B6, 0x0030B6, 0x0030B5},
{0x0030B8, 0x0030B8, 0x0030B7},
{0x0030BA, 0x0030BA, 0x0030B9},
{0x0030BC, 0x0030BC, 0x0030BB},
{0x0030BE, 0x0030BE, 0x0030BD},
{0x0030C0, 0x0030C0, 0x0030BF},
{0x0030C2, 0x0030C2, 0x0030C1},
{0x0030C5, 0x0030C5, 0x0030C4},
{0x0030C7, 0x0030C7, 0x0030C6},
{0x0030C9, 0x0030C9, 0x0030C8},
{0x0030D0, 0x0030D1, 0x0030CF},
{0x0030D3, 0x0030D4, 0x0030D2},
{0x0030D6, 0x0030D7, 0x0030D5},
{0x0030D9, 0x0030DA, 0x0030D8},
{0x0030DC, 0x0030DD, 0x0030DB},
{0x0030F4, 0x0030F4, 0x0030A6},
{0x0030F7, 0x0030F7, 0x0030EF},
{0x0030F8, 0x0030F8, 0x0030F0},
{0x0030F9, 0x0030F9, 0x0030F1},
{0x0030FA, 0x0030FA, 0x0030F2},
{0x0030FE, 0x0030FE, 0x0030FD},
{0x00AC00, 0x00AE4B, 0x001100},
{0x00AE4C, 0x00B097, 0x001101},
{0x00B098, 0x00B2E3, 0x001102},
{0x00B2E4, 0x00B52F, 0x001103},
{0x00B530, 0x00B77B, 0x001104},
{0x00B77C, 0x00B9C7, 0x001105},
{0x00B9C8, 0x00BC13, 0x001106},
{0x00BC14, 0x00BE5F, 0x001107},
{0x00BE60, 0x00C0AB, 0x001108},
{0x00C0AC, 0x00C2F7, 0x001109},
{0x00C2F8, 0x00C543, 0x00110A},
{0x00C544, 0x00C78F, 0x00110B},
{0x00C790, 0x00C9DB, 0x00110C},
{0x00C9DC, 0x00CC27, 0x00110D},
{0x00CC28, 0x00CE73, 0x00110E},
{0x00CE74, 0x00D0BF, 0x00110F},
{0x00D0C0, 0x00D30B, 0x001110},
{0x00D30C, 0x00D557, 0x001111},
{0x00D558, 0x00D7A3, 0x001112},
{0x00F900, 0x00F900, 0x008C48},
{0x00F901, 0x00F901, 0x0066F4},
{0x00F902, 0x00F902, 0x008ECA},
{0x00F903, 0x00F903, 0x008CC8},
{0x00F904, 0x00F904, 0x006ED1},
{0x00F905, 0x00F905, 0x004E32},
{0x00F906, 0x00F906, 0x0053E5},
{0x00F907, 0x00F908, 0x009F9C},
{0x00F909, 0x00F909, 0x005951},
{0x00F90A, 0x00F90A, 0x0091D1},
{0x00F90B, 0x00F90B, 0x005587},
{0x00F90C, 0x00F90C, 0x005948},
{0x00F90D, 0x00F90D, 0x0061F6},
{0x00F90E, 0x00F90E, 0x007669},
{0x00F90F, 0x00F90F, 0x007F85},
{0x00F910, 0x00F910, 0x00863F},
{0x00F911, 0x00F911, 0x0087BA},
{0x00F912, 0x00F912, 0x0088F8},
{0x00F913, 0x00F913, 0x00908F},
{0x00F914, 0x00F914, 0x006A02},
{0x00F915, 0x00F915, 0x006D1B},
{0x00F916, 0x00F916, 0x0070D9},
{0x00F917, 0x00F917, 0x0073DE},
{0x00F918, 0x00F918, 0x00843D},
{0x00F919, 0x00F919, 0x00916A},
{0x00F91A, 0x00F91A, 0x0099F1},
{0x00F91B, 0x00F91B, 0x004E82},
{0x00F91C, 0x00F91C, 0x005375},
{0x00F91D, 0x00F91D, 0x006B04},
{0x00F91E, 0x00F91E, 0x00721B},
{0x00F91F, 0x00F91F, 0x00862D},
{0x00F920, 0x00F920, 0x009E1E},
{0x00F921, 0x00F921, 0x005D50},
{0x00F922, 0x00F922, 0x006FEB},
{0x00F923, 0x00F923, 0x0085CD},
{0x00F924, 0x00F924, 0x008964},
{0x00F925, 0x00F925, 0x0062C9},
{0x00F926, 0x00F926, 0x0081D8},
{0x00F927, 0x00F927, 0x00881F},
{0x00F928, 0x00F928, 0x005ECA},
{0x00F929, 0x00F929, 0x006717},
{0x00F92A, 0x00F92A, 0x006D6A},
{0x00F92B, 0x00F92B, 0x0072FC},
{0x00F92C, 0x00F92C, 0x0090CE},
{0x00F92D, 0x00F92D, 0x004F86},
{0x00F92E, 0x00F92E, 0x0051B7},
{0x00F92F, 0x00F92F, 0x0052DE},
{0x00F930, 0x00F930, 0x0064C4},
{0x00F931, 0x00F931, 0x006AD3},
{0x00F932, 0x00F932, 0x007210},
{0x00F933, 0x00F933, 0x0076E7},
{0x00F934, 0x00F934, 0x008001},
{0x00F935, 0x00F935, 0x008606},
{0x00F936, 0x00F936, 0x00865C},
{0x00F937, 0x00F937, 0x008DEF},
{0x00F938, 0x00F938, 0x009732},
{0x00F939, 0x00F939, 0x009B6F},
{0x00F93A, 0x00F93A, 0x009DFA},
{0x00F93B, 0x00F93B, 0x00788C},
{0x00F93C, 0x00F93C, 0x00797F},
{0x00F93D, 0x00F93D, 0x007DA0},
{0x00F93E, 0x00F93E, 0x0083C9},
{0x00F93F, 0x00F93F, 0x009304},
{0x00F940, 0x00F940, 0x009E7F},
{0x00F941, 0x00F941, 0x008AD6},
{0x00F942, 0x00F942, 0x0058DF},
{0x00F943, 0x00F943, 0x005F04},
{0x00F944, 0x00F944, 0x007C60},
{0x00F945, 0x00F945, 0x00807E},
{0x00F946, 0x00F946, 0x007262},
{0x00F947, 0x00F947, 0x0078CA},
{0x00F948, 0x00F948, 0x008CC2},
{0x00F949, 0x00F949, 0x0096F7},
{0x00F94A, 0x00F94A, 0x0058D8},
{0x00F94B, 0x00F94B, 0x005C62},
{0x00F94C, 0x00F94C, 0x006A13},
{0x00F94D, 0x00F94D, 0x006DDA},
{0x00F94E, 0x00F94E, 0x006F0F},
{0x00F94F, 0x00F94F, 0x007D2F},
{0x00F950, 0x00F950, 0x007E37},
{0x00F951, 0x00F951, 0x00964B},
{0x00F952, 0x00F952, 0x0052D2},
{0x00F953, 0x00F953, 0x00808B},
{0x00F954, 0x00F954, 0x0051DC},
{0x00F955, 0x00F955, 0x0051CC},
{0x00F956, 0x00F956, 0x007A1C},
{0x00F957, 0x00F957, 0x007DBE},
{0x00F958, 0x00F958, 0x0083F1},
{0x00F959, 0x00F959, 0x009675},
{0x00F95A, 0x00F95A, 0x008B80},
{0x00F95B, 0x00F95B, 0x0062CF},
{0x00F95C, 0x00F95C, 0x006A02},
{0x00F95D, 0x00F95D, 0x008AFE},
{0x00F95E, 0x00F95E, 0x004E39},
{0x00F95F, 0x00F95F, 0x005BE7},
{0x00F960, 0x00F960, 0x006012},
{0x00F961, 0x00F961, 0x007387},
{0x00F962, 0x00F962, 0x007570},
{0x00F963, 0x00F963, 0x005317},
{0x00F964, 0x00F964, 0x0078FB},
{0x00F965, 0x00F965, 0x004FBF},
{0x00F966, 0x00F966, 0x005FA9},
{0x00F967, 0x00F967, 0x004E0D},
{0x00F968, 0x00F968, 0x006CCC},
{0x00F969, 0x00F969, 0x006578},
{0x00F96A, 0x00F96A, 0x007D22},
{0x00F96B, 0x00F96B, 0x0053C3},
{0x00F96C, 0x00F96C, 0x00585E},
{0x00F96D, 0x00F96D, 0x007701},
{0x00F96E, 0x00F96E, 0x008449},
{0x00F96F, 0x00F96F, 0x008AAA},
{0x00F970, 0x00F970, 0x006BBA},
{0x00F971, 0x00F971, 0x008FB0},
{0x00F972, 0x00F972, 0x006C88},
{0x00F973, 0x00F973, 0x0062FE},
{0x00F974, 0x00F974, 0x0082E5},
{0x00F975, 0x00F975, 0x0063A0},
{0x00F976, 0x00F976, 0x007565},
{0x00F977, 0x00F977, 0x004EAE},
{0x00F978, 0x00F978, 0x005169},
{0x00F979, 0x00F979, 0x0051C9},
{0x00F97A, 0x00F97A, 0x006881},
{0x00F97B, 0x00F97B, 0x007CE7},
{0x00F97C, 0x00F97C, 0x00826F},
{0x00F97D, 0x00F97D, 0x008AD2},
{0x00F97E, 0x00F97E, 0x0091CF},
{0x00F97F, 0x00F97F, 0x0052F5},
{0x00F980, 0x00F980, 0x005442},
{0x00F981, 0x00F981, 0x005973},
{0x00F982, 0x00F982, 0x005EEC},
{0x00F983, 0x00F983, 0x0065C5},
{0x00F984, 0x00F984, 0x006FFE},
{0x00F985, 0x00F985, 0x00792A},
{0x00F986, 0x00F986, 0x0095AD},
{0x00F987, 0x00F987, 0x009A6A},
{0x00F988, 0x00F988, 0x009E97},
{0x00F989, 0x00F989, 0x009ECE},
{0x00F98A, 0x00F98A, 0x00529B},
{0x00F98B, 0x00F98B, 0x0066C6},
{0x00F98C, 0x00F98C, 0x006B77},
{0x00F98D, 0x00F98D, 0x008F62},
{0x00F98E, 0x00F98E, 0x005E74},
{0x00F98F, 0x00F98F, 0x006190},
{0x00F990, 0x00F990, 0x006200},
{0x00F991, 0x00F991, 0x00649A},
{0x00F992, 0x00F992, 0x006F23},
{0x00F993, 0x00F993, 0x007149},
{0x00F994, 0x00F994, 0x007489},
{0x00F995, 0x00F995, 0x0079CA},
{0x00F996, 0x00F996, 0x007DF4},
{0x00F997, 0x00F997, 0x00806F},
{0x00F998, 0x00F998, 0x008F26},
{0x00F999, 0x00F999, 0x0084EE},
{0x00F99A, 0x00F99A, 0x009023},
{0x00F99B, 0x00F99B, 0x00934A},
{0x00F99C, 0x00F99C, 0x005217},
{0x00F99D, 0x00F99D, 0x0052A3},
{0x00F99E, 0x00F99E, 0x0054BD},
{0x00F99F, 0x00F99F, 0x0070C8},
{0x00F9A0, 0x00F9A0, 0x0088C2},
{0x00F9A1, 0x00F9A1, 0x008AAA},
{0x00F9A2, 0x00F9A2, 0x005EC9},
{0x00F9A3, 0x00F9A3, 0x005FF5},
{0x00F9A4, 0x00F9A4, 0x00637B},
{0x00F9A5, 0x00F9A5, 0x006BAE},
{0x00F9A6, 0x00F9A6, 0x007C3E},
{0x00F9A7, 0x00F9A7, 0x007375},
{0x00F9A8, 0x00F9A8, 0x004EE4},
{0x00F9A9, 0x00F9A9, 0x0056F9},
{0x00F9AA, 0x00F9AA, 0x005BE7},
{0x00F9AB, 0x00F9AB, 0x005DBA},
{0x00F9AC, 0x00F9AC, 0x00601C},
{0x00F9AD, 0x00F9AD, 0x0073B2},
{0x00F9AE, 0x00F9AE, 0x007469},
{0x00F9AF, 0x00F9AF, 0x007F9A},
{0x00F9B0, 0x00F9B0, 0x008046},
{0x00F9B1, 0x00F9B1, 0x009234},
{0x00F9B2, 0x00F9B2, 0x0096F6},
{0x00F9B3, 0x00F9B3, 0x009748},
{0x00F9B4, 0x00F9B4, 0x009818},
{0x00F9B5, 0x00F9B5, 0x004F8B},
{0x00F9B6, 0x00F9B6, 0x0079AE},
{0x00F9B7, 0x00F9B7, 0x0091B4},
{0x00F9B8, 0x00F9B8, 0x0096B8},
{0x00F9B9, 0x00F9B9, 0x0060E1},
{0x00F9BA, 0x00F9BA, 0x004E86},
{0x00F9BB, 0x00F9BB, 0x0050DA},
{0x00F9BC, 0x00F9BC, 0x005BEE},
{0x00F9BD, 0x00F9BD, 0x005C3F},
{0x00F9BE, 0x00F9BE, 0x006599},
{0x00F9BF, 0x00F9BF, 0x006A02},
{0x00F9C0, 0x00F9C0, 0x0071CE},
{0x00F9C1, 0x00F9C1, 0x007642},
{0x00F9C2, 0x00F9C2, 0x0084FC},
{0x00F9C3, 0x00F9C3, 0x00907C},
{0x00F9C4, 0x00F9C4, 0x009F8D},
{0x00F9C5, 0x00F9C5, 0x006688},
{0x00F9C6, 0x00F9C6, 0x00962E},
{0x00F9C7, 0x00F9C7, 0x005289},
{0x00F9C8, 0x00F9C8, 0x00677B},
{0x00F9C9, 0x00F9C9, 0x0067F3},
{0x00F9CA, 0x00F9CA, 0x006D41},
{0x00F9CB, 0x00F9CB, 0x006E9C},
{0x00F9CC, 0x00F9CC, 0x007409},
{0x00F9CD, 0x00F9CD, 0x007559},
{0x00F9CE, 0x00F9CE, 0x00786B},
{0x00F9CF, 0x00F9CF, 0x007D10},
{0x00F9D0, 0x00F9D0, 0x00985E},
{0x00F9D1, 0x00F9D1, 0x00516D},
{0x00F9D2, 0x00F9D2, 0x00622E},
{0x00F9D3, 0x00F9D3, 0x009678},
{0x00F9D4, 0x00F9D4, 0x00502B},
{0x00F9D5, 0x00F9D5, 0x005D19},
{0x00F9D6, 0x00F9D6, 0x006DEA},
{0x00F9D7, 0x00F9D7, 0x008F2A},
{0x00F9D8, 0x00F9D8, 0x005F8B},
{0x00F9D9, 0x00F9D9, 0x006144},
{0x00F9DA, 0x00F9DA, 0x006817},
{0x00F9DB, 0x00F9DB, 0x007387},
{0x00F9DC, 0x00F9DC, 0x009686},
{0x00F9DD, 0x00F9DD, 0x005229},
{0x00F9DE, 0x00F9DE, 0x00540F},
{0x00F9DF, 0x00F9DF, 0x005C65},
{0x00F9E0, 0x00F9E0, 0x006613},
{0x00F9E1, 0x00F9E1, 0x00674E},
{0x00F9E2, 0x00F9E2, 0x0068A8},
{0x00F9E3, 0x00F9E3, 0x006CE5},
{0x00F9E4, 0x00F9E4, 0x007406},
{0x00F9E5, 0x00F9E5, 0x0075E2},
{0x00F9E6, 0x00F9E6, 0x007F79},
{0x00F9E7, 0x00F9E7, 0x0088CF},
{0x00F9E8, 0x00F9E8, 0x0088E1},
{0x00F9E9, 0x00F9E9, 0x0091CC},
{0x00F9EA, 0x00F9EA, 0x0096E2},
{0x00F9EB, 0x00F9EB, 0x00533F},
{0x00F9EC, 0x00F9EC, 0x006EBA},
{0x00F9ED, 0x00F9ED, 0x00541D},
{0x00F9EE, 0x00F9EE, 0x0071D0},
{0x00F9EF, 0x00F9EF, 0x007498},
{0x00F9F0, 0x00F9F0, 0x0085FA},
{0x00F9F1, 0x00F9F1, 0x0096A3},
{0x00F9F2, 0x00F9F2, 0x009C57},
{0x00F9F3, 0x00F9F3, 0x009E9F},
{0x00F9F4, 0x00F9F4, 0x006797},
{0x00F9F5, 0x00F9F5, 0x006DCB},
{0x00F9F6, 0x00F9F6, 0x0081E8},
{0x00F9F7, 0x00F9F7, 0x007ACB},
{0x00F9F8, 0x00F9F8, 0x007B20},
{0x00F9F9, 0x00F9F9, 0x007C92},
{0x00F9FA, 0x00F9FA, 0x0072C0},
{0x00F9FB, 0x00F9FB, 0x007099},
{0x00F9FC, 0x00F9FC, 0x008B58},
{0x00F9FD, 0x00F9FD, 0x004EC0},
{0x00F9FE, 0x00F9FE, 0x008336},
{0x00F9FF, 0x00F9FF, 0x00523A},
{0x00FA00, 0x00FA00, 0x005207},
{0x00FA01, 0x00FA01, 0x005EA6},
{0x00FA02, 0x00FA02, 0x0062D3},
{0x00FA03, 0x00FA03, 0x007CD6},
{0x00FA04, 0x00FA04, 0x005B85},
{0x00FA05, 0x00FA05, 0x006D1E},
{0x00FA06, 0x00FA06, 0x0066B4},
{0x00FA07, 0x00FA07, 0x008F3B},
{0x00FA08, 0x00FA08, 0x00884C},
{0x00FA09, 0x00FA09, 0x00964D},
{0x00FA0A, 0x00FA0A, 0x00898B},
{0x00FA0B, 0x00FA0B, 0x005ED3},
{0x00FA0C, 0x00FA0C, 0x005140},
{0x00FA0D, 0x00FA0D, 0x0055C0},
{0x00FA10, 0x00FA10, 0x00585A},
{0x00FA12, 0x00FA12, 0x006674},
{0x00FA15, 0x00FA15, 0x0051DE},
{0x00FA16, 0x00FA16, 0x00732A},
{0x00FA17, 0x00FA17, 0x0076CA},
{0x00FA18, 0x00FA18, 0x00793C},
{0x00FA19, 0x00FA19, 0x00795E},
{0x00FA1A, 0x00FA1A, 0x007965},
{0x00FA1B, 0x00FA1B, 0x00798F},
{0x00FA1C, 0x00FA1C, 0x009756},
{0x00FA1D, 0x00FA1D, 0x007CBE},
{0x00FA1E, 0x00FA1E, 0x007FBD},
{0x00FA20, 0x00FA20, 0x008612},
{0x00FA22, 0x00FA22, 0x008AF8},
{0x00FA25, 0x00FA25, 0x009038},
{0x00FA26, 0x00FA26, 0x0090FD},
{0x00FA2A, 0x00FA2A, 0x0098EF},
{0x00FA2B, 0x00FA2B, 0x0098FC},
{0x00FA2C, 0x00FA2C, 0x009928},
{0x00FA2D, 0x00FA2D, 0x009DB4},
{0x00FA2E, 0x00FA2E, 0x0090DE},
{0x00FA2F, 0x00FA2F, 0x0096B7},
{0x00FA30, 0x00FA30, 0x004FAE},
{0x00FA31, 0x00FA31, 0x0050E7},
{0x00FA32, 0x00FA32, 0x00514D},
{0x00FA33, 0x00FA33, 0x0052C9},
{0x00FA34, 0x00FA34, 0x0052E4},
{0x00FA35, 0x00FA35, 0x005351},
{0x00FA36, 0x00FA36, 0x00559D},
{0x00FA37, 0x00FA37, 0x005606},
{0x00FA38, 0x00FA38, 0x005668},
{0x00FA39, 0x00FA39, 0x005840},
{0x00FA3A, 0x00FA3A, 0x0058A8},
{0x00FA3B, 0x00FA3B, 0x005C64},
{0x00FA3C, 0x00FA3C, 0x005C6E},
{0x00FA3D, 0x00FA3D, 0x006094},
{0x00FA3E, 0x00FA3E, 0x006168},
{0x00FA3F, 0x00FA3F, 0x00618E},
{0x00FA40, 0x00FA40, 0x0061F2},
{0x00FA41, 0x00FA41, 0x00654F},
{0x00FA42, 0x00FA42, 0x0065E2},
{0x00FA43, 0x00FA43, 0x006691},
{0x00FA44, 0x00FA44, 0x006885},
{0x00FA45, 0x00FA45, 0x006D77},
{0x00FA46, 0x00FA46, 0x006E1A},
{0x00FA47, 0x00FA47, 0x006F22},
{0x00FA48, 0x00FA48, 0x00716E},
{0x00FA49, 0x00FA49, 0x00722B},
{0x00FA4A, 0x00FA4A, 0x007422},
{0x00FA4B, 0x00FA4B, 0x007891},
{0x00FA4C, 0x00FA4C, 0x00793E},
{0x00FA4D, 0x00FA4D, 0x007949},
{0x00FA4E, 0x00FA4E, 0x007948},
{0x00FA4F, 0x00FA4F, 0x007950},
{0x00FA50, 0x00FA50, 0x007956},
{0x00FA51, 0x00FA51, 0x00795D},
{0x00FA52, 0x00FA52, 0x00798D},
{0x00FA53, 0x00FA53, 0x00798E},
{0x00FA54, 0x00FA54, 0x007A40},
{0x00FA55, 0x00FA55, 0x007A81},
{0x00FA56, 0x00FA56, 0x007BC0},
{0x00FA57, 0x00FA57, 0x007DF4},
{0x00FA58, 0x00FA58, 0x007E09},
{0x00FA59, 0x00FA59, 0x007E41},
{0x00FA5A, 0x00FA5A, 0x007F72},
{0x00FA5B, 0x00FA5B, 0x008005},
{0x00FA5C, 0x00FA5C, 0x0081ED},
{0x00FA5D, 0x00FA5E, 0x008279},
{0x00FA5F, 0x00FA5F, 0x008457},
{0x00FA60, 0x00FA60, 0x008910},
{0x00FA61, 0x00FA61, 0x008996},
{0x00FA62, 0x00FA62, 0x008B01},
{0x00FA63, 0x00FA63, 0x008B39},
{0x00FA64, 0x00FA64, 0x008CD3},
{0x00FA65, 0x00FA65, 0x008D08},
{0x00FA66, 0x00FA66, 0x008FB6},
{0x00FA67, 0x00FA67, 0x009038},
{0x00FA68, 0x00FA68, 0x0096E3},
{0x00FA69, 0x00FA69, 0x0097FF},
{0x00FA6A, 0x00FA6A, 0x00983B},
{0x00FA6B, 0x00FA6B, 0x006075},
{0x00FA6C, 0x00FA6C, 0x0242EE},
{0x00FA6D, 0x00FA6D, 0x008218},
{0x00FA70, 0x00FA70, 0x004E26},
{0x00FA71, 0x00FA71, 0x0051B5},
{0x00FA72, 0x00FA72, 0x005168},
{0x00FA73, 0x00FA73, 0x004F80},
{0x00FA74, 0x00FA74, 0x005145},
{0x00FA75, 0x00FA75, 0x005180},
{0x00FA76, 0x00FA76, 0x0052C7},
{0x00FA77, 0x00FA77, 0x0052FA},
{0x00FA78, 0x00FA78, 0x00559D},
{0x00FA79, 0x00FA79, 0x005555},
{0x00FA7A, 0x00FA7A, 0x005599},
{0x00FA7B, 0x00FA7B, 0x0055E2},
{0x00FA7C, 0x00FA7C, 0x00585A},
{0x00FA7D, 0x00FA7D, 0x0058B3},
{0x00FA7E, 0x00FA7E, 0x005944},
{0x00FA7F, 0x00FA7F, 0x005954},
{0x00FA80, 0x00FA80, 0x005A62},
{0x00FA81, 0x00FA81, 0x005B28},
{0x00FA82, 0x00FA82, 0x005ED2},
{0x00FA83, 0x00FA83, 0x005ED9},
{0x00FA84, 0x00FA84, 0x005F69},
{0x00FA85, 0x00FA85, 0x005FAD},
{0x00FA86, 0x00FA86, 0x0060D8},
{0x00FA87, 0x00FA87, 0x00614E},
{0x00FA88, 0x00FA88, 0x006108},
{0x00FA89, 0x00FA89, 0x00618E},
{0x00FA8A, 0x00FA8A, 0x006160},
{0x00FA8B, 0x00FA8B, 0x0061F2},
{0x00FA8C, 0x00FA8C, 0x006234},
{0x00FA8D, 0x00FA8D, 0x0063C4},
{0x00FA8E, 0x00FA8E, 0x00641C},
{0x00FA8F, 0x00FA8F, 0x006452},
{0x00FA90, 0x00FA90, 0x006556},
{0x00FA91, 0x00FA91, 0x006674},
{0x00FA92, 0x00FA92, 0x006717},
{0x00FA93, 0x00FA93, 0x00671B},
{0x00FA94, 0x00FA94, 0x006756},
{0x00FA95, 0x00FA95, 0x006B79},
{0x00FA96, 0x00FA96, 0x006BBA},
{0x00FA97, 0x00FA97, 0x006D41},
{0x00FA98, 0x00FA98, 0x006EDB},
{0x00FA99, 0x00FA99, 0x006ECB},
{0x00FA9A, 0x00FA9A, 0x006F22},
{0x00FA9B, 0x00FA9B, 0x00701E},
{0x00FA9C, 0x00FA9C, 0x00716E},
{0x00FA9D, 0x00FA9D, 0x0077A7},
{0x00FA9E, 0x00FA9E, 0x007235},
{0x00FA9F, 0x00FA9F, 0x0072AF},
{0x00FAA0, 0x00FAA0, 0x00732A},
{0x00FAA1, 0x00FAA1, 0x007471},
{0x00FAA2, 0x00FAA2, 0x007506},
{0x00FAA3, 0x00FAA3, 0x00753B},
{0x00FAA4, 0x00FAA4, 0x00761D},
{0x00FAA5, 0x00FAA5, 0x00761F},
{0x00FAA6, 0x00FAA6, 0x0076CA},
{0x00FAA7, 0x00FAA7, 0x0076DB},
{0x00FAA8, 0x00FAA8, 0x0076F4},
{0x00FAA9, 0x00FAA9, 0x00774A},
{0x00FAAA, 0x00FAAA, 0x007740},
{0x00FAAB, 0x00FAAB, 0x0078CC},
{0x00FAAC, 0x00FAAC, 0x007AB1},
{0x00FAAD, 0x00FAAD, 0x007BC0},
{0x00FAAE, 0x00FAAE, 0x007C7B},
{0x00FAAF, 0x00FAAF, 0x007D5B},
{0x00FAB0, 0x00FAB0, 0x007DF4},
{0x00FAB1, 0x00FAB1, 0x007F3E},
{0x00FAB2, 0x00FAB2, 0x008005},
{0x00FAB3, 0x00FAB3, 0x008352},
{0x00FAB4, 0x00FAB4, 0x0083EF},
{0x00FAB5, 0x00FAB5, 0x008779},
{0x00FAB6, 0x00FAB6, 0x008941},
{0x00FAB7, 0x00FAB7, 0x008986},
{0x00FAB8, 0x00FAB8, 0x008996},
{0x00FAB9, 0x00FAB9, 0x008ABF},
{0x00FABA, 0x00FABA, 0x008AF8},
{0x00FABB, 0x00FABB, 0x008ACB},
{0x00FABC, 0x00FABC, 0x008B01},
{0x00FABD, 0x00FABD, 0x008AFE},
{0x00FABE, 0x00FABE, 0x008AED},
{0x00FABF, 0x00FABF, 0x008B39},
{0x00FAC0, 0x00FAC0, 0x008B8A},
{0x00FAC1, 0x00FAC1, 0x008D08},
{0x00FAC2, 0x00FAC2, 0x008F38},
{0x00FAC3, 0x00FAC3, 0x009072},
{0x00FAC4, 0x00FAC4, 0x009199},
{0x00FAC5, 0x00FAC5, 0x009276},
{0x00FAC6, 0x00FAC6, 0x00967C},
{0x00FAC7, 0x00FAC7, 0x0096E3},
{0x00FAC8, 0x00FAC8, 0x009756},
{0x00FAC9, 0x00FAC9, 0x0097DB},
{0x00FACA, 0x00FACA, 0x0097FF},
{0x00FACB, 0x00FACB, 0x00980B},
{0x00FACC, 0x00FACC, 0x00983B},
{0x00FACD, 0x00FACD, 0x009B12},
{0x00FACE, 0x00FACE, 0x009F9C},
{0x00FACF, 0x00FACF, 0x02284A},
{0x00FAD0, 0x00FAD0, 0x022844},
{0x00FAD1, 0x00FAD1, 0x0233D5},
{0x00FAD2, 0x00FAD2, 0x003B9D},
{0x00FAD3, 0x00FAD3, 0x004018},
{0x00FAD4, 0x00FAD4, 0x004039},
{0x00FAD5, 0x00FAD5, 0x025249},
{0x00FAD6, 0x00FAD6, 0x025CD0},
{0x00FAD7, 0x00FAD7, 0x027ED3},
{0x00FAD8, 0x00FAD8, 0x009F43},
{0x00FAD9, 0x00FAD9, 0x009F8E},
{0x00FB1D, 0x00FB1D, 0x0005D9},
{0x00FB1F, 0x00FB1F, 0x0005F2},
{0x00FB2A, 0x00FB2D, 0x0005E9},
{0x00FB2E, 0x00FB30, 0x0005D0},
{0x00FB31, 0x00FB31, 0x0005D1},
{0x00FB32, 0x00FB32, 0x0005D2},
{0x00FB33, 0x00FB33, 0x0005D3},
{0x00FB34, 0x00FB34, 0x0005D4},
{0x00FB35, 0x00FB35, 0x0005D5},
{0x00FB36, 0x00FB36, 0x0005D6},
{0x00FB38, 0x00FB38, 0x0005D8},
{0x00FB39, 0x00FB39, 0x0005D9},
{0x00FB3A, 0x00FB3A, 0x0005DA},
{0x00FB3B, 0x00FB3B, 0x0005DB},
{0x00FB3C, 0x00FB3C, 0x0005DC},
{0x00FB3E, 0x00FB3E, 0x0005DE},
{0x00FB40, 0x00FB40, 0x0005E0},
{0x00FB41, 0x00FB41, 0x0005E1},
{0x00FB43, 0x00FB43, 0x0005E3},
{0x00FB44, 0x00FB44, 0x0005E4},
{0x00FB46, 0x00FB46, 0x0005E6},
{0x00FB47, 0x00FB47, 0x0005E7},
{0x00FB48, 0x00FB48, 0x0005E8},
{0x00FB49, 0x00FB49, 0x0005E9},
{0x00FB4A, 0x00FB4A, 0x0005EA},
{0x00FB4B, 0x00FB4B, 0x0005D5},
{0x00FB4C, 0x00FB4C, 0x0005D1},
{0x00FB4D, 0x00FB4D, 0x0005DB},
{0x00FB4E, 0x00FB4E, 0x0005E4},
{0x01109A, 0x01109A, 0x011099},
{0x01109C, 0x01109C, 0x01109B},
{0x0110AB, 0x0110AB, 0x0110A5},
{0x01112E, 0x01112E, 0x011131},
{0x01112F, 0x01112F, 0x011132},
{0x01134B, 0x01134C, 0x011347},
{0x0114BB, 0x0114BC, 0x0114B9},
{0x0114BE, 0x0114BE, 0x0114B9},
{0x0115BA, 0x0115BA, 0x0115B8},
{0x0115BB, 0x0115BB, 0x0115B9},
{0x011938, 0x011938, 0x011935},
{0x01D15E, 0x01D15E, 0x01D157},
{0x01D15F, 0x01D164, 0x01D158},
{0x01D1BB, 0x01D1BB, 0x01D1B9},
{0x01D1BC, 0x01D1BC, 0x01D1BA},
{0x01D1BD, 0x01D1BD, 0x01D1B9},
{0x01D1BE, 0x01D1BE, 0x01D1BA},
{0x01D1BF, 0x01D1BF, 0x01D1B9},
{0x01D1C0, 0x01D1C0, 0x01D1BA},
{0x02F800, 0x02F800, 0x004E3D},
{0x02F801, 0x02F801, 0x004E38},
{0x02F802, 0x02F802, 0x004E41},
{0x02F803, 0x02F803, 0x020122},
{0x02F804, 0x02F804, 0x004F60},
{0x02F805, 0x02F805, 0x004FAE},
{0x02F806, 0x02F806, 0x004FBB},
{0x02F807, 0x02F807, 0x005002},
{0x02F808, 0x02F808, 0x00507A},
{0x02F809, 0x02F809, 0x005099},
{0x02F80A, 0x02F80A, 0x0050E7},
{0x02F80B, 0x02F80B, 0x0050CF},
{0x02F80C, 0x02F80C, 0x00349E},
{0x02F80D, 0x02F80D, 0x02063A},
{0x02F80E, 0x02F80E, 0x00514D},
{0x02F80F, 0x02F80F, 0x005154},
{0x02F810, 0x02F810, 0x005164},
{0x02F811, 0x02F811, 0x005177},
{0x02F812, 0x02F812, 0x02051C},
{0x02F813, 0x02F813, 0x0034B9},
{0x02F814, 0x02F814, 0x005167},
{0x02F815, 0x02F815, 0x00518D},
{0x02F816, 0x02F816, 0x02054B},
{0x02F817, 0x02F817, 0x005197},
{0x02F818, 0x02F818, 0x0051A4},
{0x02F819, 0x02F819, 0x004ECC},
{0x02F81A, 0x02F81A, 0x0051AC},
{0x02F81B, 0x02F81B, 0x0051B5},
{0x02F81C, 0x02F81C, 0x0291DF},
{0x02F81D, 0x02F81D, 0x0051F5},
{0x02F81E, 0x02F81E, 0x005203},
{0x02F81F, 0x02F81F, 0x0034DF},
{0x02F820, 0x02F820, 0x00523B},
{0x02F821, 0x02F821, 0x005246},
{0x02F822, 0x02F822, 0x005272},
{0x02F823, 0x02F823, 0x005277},
{0x02F824, 0x02F824, 0x003515},
{0x02F825, 0x02F825, 0x0052C7},
{0x02F826, 0x02F826, 0x0052C9},
{0x02F827, 0x02F827, 0x0052E4},
{0x02F828, 0x02F828, 0x0052FA},
{0x02F829, 0x02F829, 0x005305},
{0x02F82A, 0x02F82A, 0x005306},
{0x02F82B, 0x02F82B, 0x005317},
{0x02F82C, 0x02F82C, 0x005349},
{0x02F82D, 0x02F82D, 0x005351},
{0x02F82E, 0x02F82E, 0x00535A},
{0x02F82F, 0x02F82F, 0x005373},
{0x02F830, 0x02F830, 0x00537D},
{0x02F831, 0x02F833, 0x00537F},
{0x02F834, 0x02F834, 0x020A2C},
{0x02F835, 0x02F835, 0x007070},
{0x02F836, 0x02F836, 0x0053CA},
{0x02F837, 0x02F837, 0x0053DF},
{0x02F838, 0x02F838, 0x020B63},
{0x02F839, 0x02F839, 0x0053EB},
{0x02F83A, 0x02F83A, 0x0053F1},
{0x02F83B, 0x02F83B, 0x005406},
{0x02F83C, 0x02F83C, 0x00549E},
{0x02F83D, 0x02F83D, 0x005438},
{0x02F83E, 0x02F83E, 0x005448},
{0x02F83F, 0x02F83F, 0x005468},
{0x02F840, 0x02F840, 0x0054A2},
{0x02F841, 0x02F841, 0x0054F6},
{0x02F842, 0x02F842, 0x005510},
{0x02F843, 0x02F843, 0x005553},
{0x02F844, 0x02F844, 0x005563},
{0x02F845, 0x02F846, 0x005584},
{0x02F847, 0x02F847, 0x005599},
{0x02F848, 0x02F848, 0x0055AB},
{0x02F849, 0x02F849, 0x0055B3},
{0x02F84A, 0x02F84A, 0x0055C2},
{0x02F84B, 0x02F84B, 0x005716},
{0x02F84C, 0x02F84C, 0x005606},
{0x02F84D, 0x02F84D, 0x005717},
{0x02F84E, 0x02F84E, 0x005651},
{0x02F84F, 0x02F84F, 0x005674},
{0x02F850, 0x02F850, 0x005207},
{0x02F851, 0x02F851, 0x0058EE},
{0x02F852, 0x02F852, 0x0057CE},
{0x02F853, 0x02F853, 0x0057F4},
{0x02F854, 0x02F854, 0x00580D},
{0x02F855, 0x02F855, 0x00578B},
{0x02F856, 0x02F856, 0x005832},
{0x02F857, 0x02F857, 0x005831},
{0x02F858, 0x02F858, 0x0058AC},
{0x02F859, 0x02F859, 0x0214E4},
{0x02F85A, 0x02F85A, 0x0058F2},
{0x02F85B, 0x02F85B, 0x0058F7},
{0x02F85C, 0x02F85C, 0x005906},
{0x02F85D, 0x02F85D, 0x00591A},
{0x02F85E, 0x02F85E, 0x005922},
{0x02F85F, 0x02F85F, 0x005962},
{0x02F860, 0x02F860, 0x0216A8},
{0x02F861, 0x02F861, 0x0216EA},
{0x02F862, 0x02F862, 0x0059EC},
{0x02F863, 0x02F863, 0x005A1B},
{0x02F864, 0x02F864, 0x005A27},
{0x02F865, 0x02F865, 0x0059D8},
{0x02F866, 0x02F866, 0x005A66},
{0x02F867, 0x02F867, 0x0036EE},
{0x02F868, 0x02F868, 0x0036FC},
{0x02F869, 0x02F869, 0x005B08},
{0x02F86A, 0x02F86B, 0x005B3E},
{0x02F86C, 0x02F86C, 0x0219C8},
{0x02F86D, 0x02F86D, 0x005BC3},
{0x02F86E, 0x02F86E, 0x005BD8},
{0x02F86F, 0x02F86F, 0x005BE7},
{0x02F870, 0x02F870, 0x005BF3},
{0x02F871, 0x02F871, 0x021B18},
{0x02F872, 0x02F872, 0x005BFF},
{0x02F873, 0x02F873, 0x005C06},
{0x02F874, 0x02F874, 0x005F53},
{0x02F875, 0x02F875, 0x005C22},
{0x02F876, 0x02F876, 0x003781},
{0x02F877, 0x02F877, 0x005C60},
{0x02F878, 0x02F878, 0x005C6E},
{0x02F879, 0x02F879, 0x005CC0},
{0x02F87A, 0x02F87A, 0x005C8D},
{0x02F87B, 0x02F87B, 0x021DE4},
{0x02F87C, 0x02F87C, 0x005D43},
{0x02F87D, 0x02F87D, 0x021DE6},
{0x02F87E, 0x02F87E, 0x005D6E},
{0x02F87F, 0x02F87F, 0x005D6B},
{0x02F880, 0x02F880, 0x005D7C},
{0x02F881, 0x02F881, 0x005DE1},
{0x02F882, 0x02F882, 0x005DE2},
{0x02F883, 0x02F883, 0x00382F},
{0x02F884, 0x02F884, 0x005DFD},
{0x02F885, 0x02F885, 0x005E28},
{0x02F886, 0x02F886, 0x005E3D},
{0x02F887, 0x02F887, 0x005E69},
{0x02F888, 0x02F888, 0x003862},
{0x02F889, 0x02F889, 0x022183},
{0x02F88A, 0x02F88A, 0x00387C},
{0x02F88B, 0x02F88B, 0x005EB0},
{0x02F88C, 0x02F88C, 0x005EB3},
{0x02F88D, 0x02F88D, 0x005EB6},
{0x02F88E, 0x02F88E, 0x005ECA},
{0x02F88F, 0x02F88F, 0x02A392},
{0x02F890, 0x02F890, 0x005EFE},
{0x02F891, 0x02F892, 0x022331},
{0x02F893, 0x02F893, 0x008201},
{0x02F894, 0x02F895, 0x005F22},
{0x02F896, 0x02F896, 0x0038C7},
{0x02F897, 0x02F897, 0x0232B8},
{0x02F898, 0x02F898, 0x0261DA},
{0x02F899, 0x02F899, 0x005F62},
{0x02F89A, 0x02F89A, 0x005F6B},
{0x02F89B, 0x02F89B, 0x0038E3},
{0x02F89C, 0x02F89C, 0x005F9A},
{0x02F89D, 0x02F89D, 0x005FCD},
{0x02F89E, 0x02F89E, 0x005FD7},
{0x02F89F, 0x02F89F, 0x005FF9},
{0x02F8A0, 0x02F8A0, 0x006081},
{0x02F8A1, 0x02F8A1, 0x00393A},
{0x02F8A2, 0x02F8A2, 0x00391C},
{0x02F8A3, 0x02F8A3, 0x006094},
{0x02F8A4, 0x02F8A4, 0x0226D4},
{0x02F8A5, 0x02F8A5, 0x0060C7},
{0x02F8A6, 0x02F8A6, 0x006148},
{0x02F8A7, 0x02F8A7, 0x00614C},
{0x02F8A8, 0x02F8A8, 0x00614E},
{0x02F8A9, 0x02F8A9, 0x00614C},
{0x02F8AA, 0x02F8AA, 0x00617A},
{0x02F8AB, 0x02F8AB, 0x00618E},
{0x02F8AC, 0x02F8AC, 0x0061B2},
{0x02F8AD, 0x02F8AD, 0x0061A4},
{0x02F8AE, 0x02F8AE, 0x0061AF},
{0x02F8AF, 0x02F8AF, 0x0061DE},
{0x02F8B0, 0x02F8B0, 0x0061F2},
{0x02F8B1, 0x02F8B1, 0x0061F6},
{0x02F8B2, 0x02F8B2, 0x006210},
{0x02F8B3, 0x02F8B3, 0x00621B},
{0x02F8B4, 0x02F8B4, 0x00625D},
{0x02F8B5, 0x02F8B5, 0x0062B1},
{0x02F8B6, 0x02F8B6, 0x0062D4},
{0x02F8B7, 0x02F8B7, 0x006350},
{0x02F8B8, 0x02F8B8, 0x022B0C},
{0x02F8B9, 0x02F8B9, 0x00633D},
{0x02F8BA, 0x02F8BA, 0x0062FC},
{0x02F8BB, 0x02F8BB, 0x006368},
{0x02F8BC, 0x02F8BC, 0x006383},
{0x02F8BD, 0x02F8BD, 0x0063E4},
{0x02F8BE, 0x02F8BE, 0x022BF1},
{0x02F8BF, 0x02F8BF, 0x006422},
{0x02F8C0, 0x02F8C0, 0x0063C5},
{0x02F8C1, 0x02F8C1, 0x0063A9},
{0x02F8C2, 0x02F8C2, 0x003A2E},
{0x02F8C3, 0x02F8C3, 0x006469},
{0x02F8C4, 0x02F8C4, 0x00647E},
{0x02F8C5, 0x02F8C5, 0x00649D},
{0x02F8C6, 0x02F8C6, 0x006477},
{0x02F8C7, 0x02F8C7, 0x003A6C},
{0x02F8C8, 0x02F8C8, 0x00654F},
{0x02F8C9, 0x02F8C9, 0x00656C},
{0x02F8CA, 0x02F8CA, 0x02300A},
{0x02F8CB, 0x02F8CB, 0x0065E3},
{0x02F8CC, 0x02F8CC, 0x0066F8},
{0x02F8CD, 0x02F8CD, 0x006649},
{0x02F8CE, 0x02F8CE, 0x003B19},
{0x02F8CF, 0x02F8CF, 0x006691},
{0x02F8D0, 0x02F8D0, 0x003B08},
{0x02F8D1, 0x02F8D1, 0x003AE4},
{0x02F8D2, 0x02F8D2, 0x005192},
{0x02F8D3, 0x02F8D3, 0x005195},
{0x02F8D4, 0x02F8D4, 0x006700},
{0x02F8D5, 0x02F8D5, 0x00669C},
{0x02F8D6, 0x02F8D6, 0x0080AD},
{0x02F8D7, 0x02F8D7, 0x0043D9},
{0x02F8D8, 0x02F8D8, 0x006717},
{0x02F8D9, 0x02F8D9, 0x00671B},
{0x02F8DA, 0x02F8DA, 0x006721},
{0x02F8DB, 0x02F8DB, 0x00675E},
{0x02F8DC, 0x02F8DC, 0x006753},
{0x02F8DD, 0x02F8DD, 0x0233C3},
{0x02F8DE, 0x02F8DE, 0x003B49},
{0x02F8DF, 0x02F8DF, 0x0067FA},
{0x02F8E0, 0x02F8E0, 0x006785},
{0x02F8E1, 0x02F8E1, 0x006852},
{0x02F8E2, 0x02F8E2, 0x006885},
{0x02F8E3, 0x02F8E3, 0x02346D},
{0x02F8E4, 0x02F8E4, 0x00688E},
{0x02F8E5, 0x02F8E5, 0x00681F},
{0x02F8E6, 0x02F8E6, 0x006914},
{0x02F8E7, 0x02F8E7, 0x003B9D},
{0x02F8E8, 0x02F8E8, 0x006942},
{0x02F8E9, 0x02F8E9, 0x0069A3},
{0x02F8EA, 0x02F8EA, 0x0069EA},
{0x02F8EB, 0x02F8EB, 0x006AA8},
{0x02F8EC, 0x02F8EC, 0x0236A3},
{0x02F8ED, 0x02F8ED, 0x006ADB},
{0x02F8EE, 0x02F8EE, 0x003C18},
{0x02F8EF, 0x02F8EF, 0x006B21},
{0x02F8F0, 0x02F8F0, 0x0238A7},
{0x02F8F1, 0x02F8F1, 0x006B54},
{0x02F8F2, 0x02F8F2, 0x003C4E},
{0x02F8F3, 0x02F8F3, 0x006B72},
{0x02F8F4, 0x02F8F4, 0x006B9F},
{0x02F8F5, 0x02F8F5, 0x006BBA},
{0x02F8F6, 0x02F8F6, 0x006BBB},
{0x02F8F7, 0x02F8F7, 0x023A8D},
{0x02F8F8, 0x02F8F8, 0x021D0B},
{0x02F8F9, 0x02F8F9, 0x023AFA},
{0x02F8FA, 0x02F8FA, 0x006C4E},
{0x02F8FB, 0x02F8FB, 0x023CBC},
{0x02F8FC, 0x02F8FC, 0x006CBF},
{0x02F8FD, 0x02F8FD, 0x006CCD},
{0x02F8FE, 0x02F8FE, 0x006C67},
{0x02F8FF, 0x02F8FF, 0x006D16},
{0x02F900, 0x02F900, 0x006D3E},
{0x02F901, 0x02F901, 0x006D77},
{0x02F902, 0x02F902, 0x006D41},
{0x02F903, 0x02F903, 0x006D69},
{0x02F904, 0x02F904, 0x006D78},
{0x02F905, 0x02F905, 0x006D85},
{0x02F906, 0x02F906, 0x023D1E},
{0x02F907, 0x02F907, 0x006D34},
{0x02F908, 0x02F908, 0x006E2F},
{0x02F909, 0x02F909, 0x006E6E},
{0x02F90A, 0x02F90A, 0x003D33},
{0x02F90B, 0x02F90B, 0x006ECB},
{0x02F90C, 0x02F90C, 0x006EC7},
{0x02F90D, 0x02F90D, 0x023ED1},
{0x02F90E, 0x02F90E, 0x006DF9},
{0x02F90F, 0x02F90F, 0x006F6E},
{0x02F910, 0x02F910, 0x023F5E},
{0x02F911, 0x02F911, 0x023F8E},
{0x02F912, 0x02F912, 0x006FC6},
{0x02F913, 0x02F913, 0x007039},
{0x02F914, 0x02F914, 0x00701E},
{0x02F915, 0x02F915, 0x00701B},
{0x02F916, 0x02F916, 0x003D96},
{0x02F917, 0x02F917, 0x00704A},
{0x02F918, 0x02F918, 0x00707D},
{0x02F919, 0x02F919, 0x007077},
{0x02F91A, 0x02F91A, 0x0070AD},
{0x02F91B, 0x02F91B, 0x020525},
{0x02F91C, 0x02F91C, 0x007145},
{0x02F91D, 0x02F91D, 0x024263},
{0x02F91E, 0x02F91E, 0x00719C},
{0x02F91F, 0x02F91F, 0x0243AB},
{0x02F920, 0x02F920, 0x007228},
{0x02F921, 0x02F921, 0x007235},
{0x02F922, 0x02F922, 0x007250},
{0x02F923, 0x02F923, 0x024608},
{0x02F924, 0x02F924, 0x007280},
{0x02F925, 0x02F925, 0x007295},
{0x02F926, 0x02F926, 0x024735},
{0x02F927, 0x02F927, 0x024814},
{0x02F928, 0x02F928, 0x00737A},
{0x02F929, 0x02F929, 0x00738B},
{0x02F92A, 0x02F92A, 0x003EAC},
{0x02F92B, 0x02F92B, 0x0073A5},
{0x02F92C, 0x02F92D, 0x003EB8},
{0x02F92E, 0x02F92E, 0x007447},
{0x02F92F, 0x02F92F, 0x00745C},
{0x02F930, 0x02F930, 0x007471},
{0x02F931, 0x02F931, 0x007485},
{0x02F932, 0x02F932, 0x0074CA},
{0x02F933, 0x02F933, 0x003F1B},
{0x02F934, 0x02F934, 0x007524},
{0x02F935, 0x02F935, 0x024C36},
{0x02F936, 0x02F936, 0x00753E},
{0x02F937, 0x02F937, 0x024C92},
{0x02F938, 0x02F938, 0x007570},
{0x02F939, 0x02F939, 0x02219F},
{0x02F93A, 0x02F93A, 0x007610},
{0x02F93B, 0x02F93B, 0x024FA1},
{0x02F93C, 0x02F93C, 0x024FB8},
{0x02F93D, 0x02F93D, 0x025044},
{0x02F93E, 0x02F93E, 0x003FFC},
{0x02F93F, 0x02F93F, 0x004008},
{0x02F940, 0x02F940, 0x0076F4},
{0x02F941, 0x02F941, 0x0250F3},
{0x02F942, 0x02F942, 0x0250F2},
{0x02F943, 0x02F943, 0x025119},
{0x02F944, 0x02F944, 0x025133},
{0x02F945, 0x02F945, 0x00771E},
{0x02F946, 0x02F947, 0x00771F},
{0x02F948, 0x02F948, 0x00774A},
{0x02F949, 0x02F949, 0x004039},
{0x02F94A, 0x02F94A, 0x00778B},
{0x02F94B, 0x02F94B, 0x004046},
{0x02F94C, 0x02F94C, 0x004096},
{0x02F94D, 0x02F94D, 0x02541D},
{0x02F94E, 0x02F94E, 0x00784E},
{0x02F94F, 0x02F94F, 0x00788C},
{0x02F950, 0x02F950, 0x0078CC},
{0x02F951, 0x02F951, 0x0040E3},
{0x02F952, 0x02F952, 0x025626},
{0x02F953, 0x02F953, 0x007956},
{0x02F954, 0x02F954, 0x02569A},
{0x02F955, 0x02F955, 0x0256C5},
{0x02F956, 0x02F956, 0x00798F},
{0x02F957, 0x02F957, 0x0079EB},
{0x02F958, 0x02F958, 0x00412F},
{0x02F959, 0x02F959, 0x007A40},
{0x02F95A, 0x02F95A, 0x007A4A},
{0x02F95B, 0x02F95B, 0x007A4F},
{0x02F95C, 0x02F95C, 0x02597C},
{0x02F95D, 0x02F95E, 0x025AA7},
{0x02F95F, 0x02F95F, 0x007AEE},
{0x02F960, 0x02F960, 0x004202},
{0x02F961, 0x02F961, 0x025BAB},
{0x02F962, 0x02F962, 0x007BC6},
{0x02F963, 0x02F963, 0x007BC9},
{0x02F964, 0x02F964, 0x004227},
{0x02F965, 0x02F965, 0x025C80},
{0x02F966, 0x02F966, 0x007CD2},
{0x02F967, 0x02F967, 0x0042A0},
{0x02F968, 0x02F968, 0x007CE8},
{0x02F969, 0x02F969, 0x007CE3},
{0x02F96A, 0x02F96A, 0x007D00},
{0x02F96B, 0x02F96B, 0x025F86},
{0x02F96C, 0x02F96C, 0x007D63},
{0x02F96D, 0x02F96D, 0x004301},
{0x02F96E, 0x02F96E, 0x007DC7},
{0x02F96F, 0x02F96F, 0x007E02},
{0x02F970, 0x02F970, 0x007E45},
{0x02F971, 0x02F971, 0x004334},
{0x02F972, 0x02F972, 0x026228},
{0x02F973, 0x02F973, 0x026247},
{0x02F974, 0x02F974, 0x004359},
{0x02F975, 0x02F975, 0x0262D9},
{0x02F976, 0x02F976, 0x007F7A},
{0x02F977, 0x02F977, 0x02633E},
{0x02F978, 0x02F978, 0x007F95},
{0x02F979, 0x02F979, 0x007FFA},
{0x02F97A, 0x02F97A, 0x008005},
{0x02F97B, 0x02F97B, 0x0264DA},
{0x02F97C, 0x02F97C, 0x026523},
{0x02F97D, 0x02F97D, 0x008060},
{0x02F97E, 0x02F97E, 0x0265A8},
{0x02F97F, 0x02F97F, 0x008070},
{0x02F980, 0x02F980, 0x02335F},
{0x02F981, 0x02F981, 0x0043D5},
{0x02F982, 0x02F982, 0x0080B2},
{0x02F983, 0x02F983, 0x008103},
{0x02F984, 0x02F984, 0x00440B},
{0x02F985, 0x02F985, 0x00813E},
{0x02F986, 0x02F986, 0x005AB5},
{0x02F987, 0x02F987, 0x0267A7},
{0x02F988, 0x02F988, 0x0267B5},
{0x02F989, 0x02F989, 0x023393},
{0x02F98A, 0x02F98A, 0x02339C},
{0x02F98B, 0x02F98B, 0x008201},
{0x02F98C, 0x02F98C, 0x008204},
{0x02F98D, 0x02F98D, 0x008F9E},
{0x02F98E, 0x02F98E, 0x00446B},
{0x02F98F, 0x02F98F, 0x008291},
{0x02F990, 0x02F990, 0x00828B},
{0x02F991, 0x02F991, 0x00829D},
{0x02F992, 0x02F992, 0x0052B3},
{0x02F993, 0x02F993, 0x0082B1},
{0x02F994, 0x02F994, 0x0082B3},
{0x02F995, 0x02F995, 0x0082BD},
{0x02F996, 0x02F996, 0x0082E6},
{0x02F997, 0x02F997, 0x026B3C},
{0x02F998, 0x02F998, 0x0082E5},
{0x02F999, 0x02F999, 0x00831D},
{0x02F99A, 0x02F99A, 0x008363},
{0x02F99B, 0x02F99B, 0x0083AD},
{0x02F99C, 0x02F99C, 0x008323},
{0x02F99D, 0x02F99D, 0x0083BD},
{0x02F99E, 0x02F99E, 0x0083E7},
{0x02F99F, 0x02F99F, 0x008457},
{0x02F9A0, 0x02F9A0, 0x008353},
{0x02F9A1, 0x02F9A1, 0x0083CA},
{0x02F9A2, 0x02F9A2, 0x0083CC},
{0x02F9A3, 0x02F9A3, 0x0083DC},
{0x02F9A4, 0x02F9A4, 0x026C36},
{0x02F9A5, 0x02F9A5, 0x026D6B},
{0x02F9A6, 0x02F9A6, 0x026CD5},
{0x02F9A7, 0x02F9A7, 0x00452B},
{0x02F9A8, 0x02F9A8, 0x0084F1},
{0x02F9A9, 0x02F9A9, 0x0084F3},
{0x02F9AA, 0x02F9AA, 0x008516},
{0x02F9AB, 0x02F9AB, 0x0273CA},
{0x02F9AC, 0x02F9AC, 0x008564},
{0x02F9AD, 0x02F9AD, 0x026F2C},
{0x02F9AE, 0x02F9AE, 0x00455D},
{0x02F9AF, 0x02F9AF, 0x004561},
{0x02F9B0, 0x02F9B0, 0x026FB1},
{0x02F9B1, 0x02F9B1, 0x0270D2},
{0x02F9B2, 0x02F9B2, 0x00456B},
{0x02F9B3, 0x02F9B3, 0x008650},
{0x02F9B4, 0x02F9B4, 0x00865C},
{0x02F9B5, 0x02F9B5, 0x008667},
{0x02F9B6, 0x02F9B6, 0x008669},
{0x02F9B7, 0x02F9B7, 0x0086A9},
{0x02F9B8, 0x02F9B8, 0x008688},
{0x02F9B9, 0x02F9B9, 0x00870E},
{0x02F9BA, 0x02F9BA, 0x0086E2},
{0x02F9BB, 0x02F9BB, 0x008779},
{0x02F9BC, 0x02F9BC, 0x008728},
{0x02F9BD, 0x02F9BD, 0x00876B},
{0x02F9BE, 0x02F9BE, 0x008786},
{0x02F9BF, 0x02F9BF, 0x0045D7},
{0x02F9C0, 0x02F9C0, 0x0087E1},
{0x02F9C1, 0x02F9C1, 0x008801},
{0x02F9C2, 0x02F9C2, 0x0045F9},
{0x02F9C3, 0x02F9C3, 0x008860},
{0x02F9C4, 0x02F9C4, 0x008863},
{0x02F9C5, 0x02F9C5, 0x027667},
{0x02F9C6, 0x02F9C6, 0x0088D7},
{0x02F9C7, 0x02F9C7, 0x0088DE},
{0x02F9C8, 0x02F9C8, 0x004635},
{0x02F9C9, 0x02F9C9, 0x0088FA},
{0x02F9CA, 0x02F9CA, 0x0034BB},
{0x02F9CB, 0x02F9CB, 0x0278AE},
{0x02F9CC, 0x02F9CC, 0x027966},
{0x02F9CD, 0x02F9CD, 0x0046BE},
{0x02F9CE, 0x02F9CE, 0x0046C7},
{0x02F9CF, 0x02F9CF, 0x008AA0},
{0x02F9D0, 0x02F9D0, 0x008AED},
{0x02F9D1, 0x02F9D1, 0x008B8A},
{0x02F9D2, 0x02F9D2, 0x008C55},
{0x02F9D3, 0x02F9D3, 0x027CA8},
{0x02F9D4, 0x02F9D4, 0x008CAB},
{0x02F9D5, 0x02F9D5, 0x008CC1},
{0x02F9D6, 0x02F9D6, 0x008D1B},
{0x02F9D7, 0x02F9D7, 0x008D77},
{0x02F9D8, 0x02F9D8, 0x027F2F},
{0x02F9D9, 0x02F9D9, 0x020804},
{0x02F9DA, 0x02F9DA, 0x008DCB},
{0x02F9DB, 0x02F9DB, 0x008DBC},
{0x02F9DC, 0x02F9DC, 0x008DF0},
{0x02F9DD, 0x02F9DD, 0x0208DE},
{0x02F9DE, 0x02F9DE, 0x008ED4},
{0x02F9DF, 0x02F9DF, 0x008F38},
{0x02F9E0, 0x02F9E0, 0x0285D2},
{0x02F9E1, 0x02F9E1, 0x0285ED},
{0x02F9E2, 0x02F9E2, 0x009094},
{0x02F9E3, 0x02F9E3, 0x0090F1},
{0x02F9E4, 0x02F9E4, 0x009111},
{0x02F9E5, 0x02F9E5, 0x02872E},
{0x02F9E6, 0x02F9E6, 0x00911B},
{0x02F9E7, 0x02F9E7, 0x009238},
{0x02F9E8, 0x02F9E8, 0x0092D7},
{0x02F9E9, 0x02F9E9, 0x0092D8},
{0x02F9EA, 0x02F9EA, 0x00927C},
{0x02F9EB, 0x02F9EB, 0x0093F9},
{0x02F9EC, 0x02F9EC, 0x009415},
{0x02F9ED, 0x02F9ED, 0x028BFA},
{0x02F9EE, 0x02F9EE, 0x00958B},
{0x02F9EF, 0x02F9EF, 0x004995},
{0x02F9F0, 0x02F9F0, 0x0095B7},
{0x02F9F1, 0x02F9F1, 0x028D77},
{0x02F9F2, 0x02F9F2, 0x0049E6},
{0x02F9F3, 0x02F9F3, 0x0096C3},
{0x02F9F4, 0x02F9F4, 0x005DB2},
{0x02F9F5, 0x02F9F5, 0x009723},
{0x02F9F6, 0x02F9F6, 0x029145},
{0x02F9F7, 0x02F9F7, 0x02921A},
{0x02F9F8, 0x02F9F8, 0x004A6E},
{0x02F9F9, 0x02F9F9, 0x004A76},
{0x02F9FA, 0x02F9FA, 0x0097E0},
{0x02F9FB, 0x02F9FB, 0x02940A},
{0x02F9FC, 0x02F9FC, 0x004AB2},
{0x02F9FD, 0x02F9FD, 0x029496},
{0x02F9FE, 0x02F9FF, 0x00980B},
{0x02FA00, 0x02FA00, 0x009829},
{0x02FA01, 0x02FA01, 0x0295B6},
{0x02FA02, 0x02FA02, 0x0098E2},
{0x02FA03, 0x02FA03, 0x004B33},
{0x02FA04, 0x02FA04, 0x009929},
{0x02FA05, 0x02FA05, 0x0099A7},
{0x02FA06, 0x02FA06, 0x0099C2},
{0x02FA07, 0x02FA07, 0x0099FE},
{0x02FA08, 0x02FA08, 0x004BCE},
{0x02FA09, 0x02FA09, 0x029B30},
{0x02FA0A, 0x02FA0A, 0x009B12},
{0x02FA0B, 0x02FA0B, 0x009C40},
{0x02FA0C, 0x02FA0C, 0x009CFD},
{0x02FA0D, 0x02FA0D, 0x004CCE},
{0x02FA0E, 0x02FA0E, 0x004CED},
{0x02FA0F, 0x02FA0F, 0x009D67},
{0x02FA10, 0x02FA10, 0x02A0CE},
{0x02FA11, 0x02FA11, 0x004CF8},
{0x02FA12, 0x02FA12, 0x02A105},
{0x02FA13, 0x02FA13, 0x02A20E},
{0x02FA14, 0x02FA14, 0x02A291},
{0x02FA15, 0x02FA15, 0x009EBB},
{0x02FA16, 0x02FA16, 0x004D56},
{0x02FA17, 0x02FA17, 0x009EF9},
{0x02FA18, 0x02FA18, 0x009EFE},
{0x02FA19, 0x02FA19, 0x009F05},
{0x02FA1A, 0x02FA1A, 0x009F0F},
{0x02FA1B, 0x02FA1B, 0x009F16},
{0x02FA1C, 0x02FA1C, 0x009F3B},
{0x02FA1D, 0x02FA1D, 0x02A600},
};
