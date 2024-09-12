/*
BERT inference in GGML

Forked with gratitude from:
https://github.com/skeskinen/bert.cpp
https://github.com/xyzhang626/embeddings.cpp
*/

#include "bert.h"
#include "ggml.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>

#define BERT_MAX_NODES 4096

const int verbosity = 1;

//
// utilities to get data from a gguf file
//

static int get_key_idx(const gguf_context * ctx, const char * key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        fprintf(stderr, "%s: key %s not found in file\n", __func__, key);
        throw;
    }

    return i;
}

static int32_t get_i32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_i32(ctx, i);
}

static uint32_t get_u32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_u32(ctx, i);
}

static float get_f32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

static std::string get_str(const gguf_context * ctx, const std::string & key, const std::string & def = "") {
    const int i = gguf_find_key(ctx, key.c_str());
    if (i == -1) {
        return def;
    }
    return gguf_get_val_str(ctx, i);
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        fprintf(stderr, "%s: unable to find tensor %s\n", __func__, name.c_str());
        throw;
    }

    return cur;
}

static std::string get_ftype(int ftype) {
    return ggml_type_name(static_cast<ggml_type>(ftype));
}

static void tensor_stats(ggml_tensor * t) {
    int32_t src0 = t->src[0] ? t->src[0]->backend : -1;
    int32_t src1 = t->src[1] ? t->src[1]->backend : -1;
    fprintf(stderr,
        "type = %s, dims = %d, shape = (%ld, %ld, %ld, %ld), backend = %d, src0 = %d, src1 = %d\n",
        ggml_type_name(t->type), ggml_n_dims(t), t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->backend, src0, src1
    );
}

//
// tokenizing
//

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

std::string strip_accents(const std::string &inputString) {
    std::string resultString;
    std::map<std::string, char> accentMap = {{"À", 'A'},{"Á", 'A'},
        {"Â", 'A'},{"Ã", 'A'},{"Ä", 'A'},{"Å", 'A'},{"à", 'a'},{"á", 'a'},
        {"â", 'a'},{"ã", 'a'},{"ä", 'a'},{"å", 'a'},{"È", 'E'},{"É", 'E'},
        {"Ê", 'E'},{"Ë", 'E'},{"è", 'e'},{"é", 'e'},{"ê", 'e'},{"ë", 'e'},
        {"Ì", 'I'},{"Í", 'I'},{"Î", 'I'},{"Ï", 'I'},{"ì", 'i'},{"í", 'i'},
        {"î", 'i'},{"ï", 'i'},{"Ò", 'O'},{"Ó", 'O'},{"Ô", 'O'},{"Õ", 'O'},
        {"Ö", 'O'},{"ò", 'o'},{"ó", 'o'},{"ô", 'o'},{"õ", 'o'},{"ö", 'o'},
        {"Ù", 'U'},{"Ú", 'U'},{"Û", 'U'},{"Ü", 'U'},{"ù", 'u'},{"ú", 'u'},
        {"û", 'u'},{"ü", 'u'},{"Ý", 'Y'},{"ý", 'y'},{"Ç", 'C'},{"ç", 'c'},
        {"Ñ", 'N'},{"ñ", 'n'},
    };

    for (size_t i = 0; i < inputString.length();)
    {
        int len = utf8_len(inputString[i]);
        std::string curChar = inputString.substr(i, len);
        auto iter = accentMap.find(curChar);
        if (iter != accentMap.end())
        {
            resultString += iter->second;
        }
        else
        {
            resultString += curChar;
        }
        i += len;
    }

    return resultString;
}

std::string bert_normalize_prompt(const std::string &text) {
    // TODO: handle chinese characters? https://github.com/huggingface/tokenizers/blob/ef5f50605ddf9f8caef1598c0e4853862b9707a7/tokenizers/src/normalizers/bert.rs#L98
    std::string text2 = strip_accents(text);
    for (size_t i = 0; i < text2.size(); i += utf8_len(text2[i]))
    {
        char c = text2[i];
        if (c >= 'A' && c <= 'Z')
            text2[i] = c - 'A' + 'a';
    }
    return text2;
}

bool is_chinese_char(const std::string& str) {
    int len = str.length();
    unsigned int codepoint = 0;
    int num_bytes = 0;
    int i = 0;
    unsigned char ch = static_cast<unsigned char>(str[i]);
    if (ch <= 0x7f) {
        codepoint = ch;
        num_bytes = 1;
    } else if ((ch >> 5) == 0x06) {
        codepoint = ch & 0x1f;
        num_bytes = 2;
    } else if ((ch >> 4) == 0x0e) {
        codepoint = ch & 0x0f;
        num_bytes = 3;
    } else if ((ch >> 3) == 0x1e) {
        codepoint = ch & 0x07;
        num_bytes = 4;
    }
    for (int j = 1; j < num_bytes; ++j) {
        if (i + j >= len) {
            return false; // incomplete UTF-8 character
        }
        unsigned char next_ch = static_cast<unsigned char>(str[i + j]);
        if ((next_ch >> 6) != 0x02) {
            return false; // invalid trailing byte
        }
        codepoint = (codepoint << 6) | (next_ch & 0x3f);
    }
    if ((codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
        (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
        (codepoint >= 0x20000 && codepoint <= 0x2A6DF) ||
        (codepoint >= 0x2A700 && codepoint <= 0x2B73F) ||
        (codepoint >= 0x2B740 && codepoint <= 0x2B81F) ||
        (codepoint >= 0x2B920 && codepoint <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
        (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||
        (codepoint >= 0x2F800 && codepoint <= 0x2FA1F) ||
        (codepoint >= 0x3000 && codepoint <= 0x303F) ||
        (codepoint >= 0xFF00 && codepoint <= 0xFFEF)) {
        return true;
    }
    return false;
}

const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_token id) {
    bert_vocab & vocab = ctx->vocab;

    auto it = vocab._id_to_token.find(id);
    if (it != vocab._id_to_token.end()) {
        return it->second.c_str();
    }
    it = vocab._id_to_subword_token.find(id);
    if (it != vocab._id_to_subword_token.end()) {
        return it->second.c_str();
    }
    return "[UNK]";
}

bert_tokens bert_tokenize(struct bert_ctx * ctx, bert_string text, uint64_t n_max_tokens) {
    const bert_vocab &vocab = ctx->vocab;
    const bert_token bos_id = vocab.bos_id;
    const bert_token eos_id = vocab.eos_id;
    const bert_token unk_id = vocab.unk_id;

    std::string normalized_text = bert_normalize_prompt(text);

    bert_tokens tokens;
    tokens.push_back(bos_id);

    for (size_t i = 0; i < normalized_text.length() && tokens.size() < n_max_tokens - 1;) {
        int utf_char_len = utf8_len(normalized_text[i]);
        std::string character = normalized_text.substr(i, utf_char_len);

        auto it = vocab.token_to_id.find(character);
        if (it != vocab.token_to_id.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_id);
        }

        i += utf_char_len;
    }

    tokens.push_back(eos_id);
    return tokens;
}

bert_string bert_detokenize(struct bert_ctx * ctx, bert_tokens tokens, bool debug = false) {
    const bert_token bos_id = ctx->vocab.bos_id;
    const bert_token eos_id = ctx->vocab.eos_id;

    bert_string str = "";
    for (const uint64_t &t : tokens) {
        if (t == bos_id || t == eos_id) {
            continue;
        }
        str += bert_vocab_id_to_token(ctx, t);
    }
    return str;
}

uint64_t bert_detokenize_c(struct bert_ctx * ctx, int32_t * tokens, char * output, uint64_t n_tokens, uint64_t n_output, bool debug) {
    bert_tokens tokens2(tokens, tokens + n_tokens);
    bert_string str = bert_detokenize(ctx, tokens2, debug);
    memcpy(output, str.c_str(), std::min<uint64_t>(n_output, str.size()));
    return str.size();
}

// c-string interface to tokenizer
uint64_t bert_tokenize_c(struct bert_ctx * ctx, const char * text, int32_t * output, uint64_t n_max_tokens) {
    bert_string str(text);
    bert_tokens tokens = bert_tokenize(ctx, str, n_max_tokens);
    memcpy(output, tokens.data(), tokens.size() * sizeof(int32_t));
    return tokens.size();
}

//
// bert model
//

int32_t bert_n_embd(bert_ctx * ctx) {
    return ctx->model.hparams.n_embd;
}

int32_t bert_n_max_tokens(bert_ctx * ctx) {
    return ctx->model.hparams.n_max_tokens;
}

//
// loading and setup
//

struct bert_ctx * bert_load_from_file(const char *fname, bool use_cpu) {
    struct ggml_context * ctx_ggml = NULL;

    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_ggml,
    };

    // open gguf file
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname, gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to load BERT model from %s. Does this file exist?\n", __func__, fname);
        return nullptr;
    }

    // get generic model info
    if (verbosity >= 1) {
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);
        const int n_kv = gguf_get_n_kv(ctx_gguf);
        const int ftype = get_u32(ctx_gguf, KEY_FTYPE);
        const int alignment = gguf_get_alignment(ctx_gguf);
        const int version = gguf_get_version(ctx_gguf);
        const std::string ftype_str = get_ftype(ftype);
        const std::string description = get_str(ctx_gguf, KEY_DESCRIPTION);
        const std::string name = get_str(ctx_gguf, KEY_NAME);

        fprintf(stderr, "\n");
        fprintf(stderr, "%s: GGUF\n", __func__);
        fprintf(stderr, "%s: model name:   %s\n", __func__, name.c_str());
        fprintf(stderr, "%s: description:  %s\n", __func__, description.c_str());
        fprintf(stderr, "%s: GGUF version: %d\n", __func__, version);
        fprintf(stderr, "%s: alignment:    %d\n", __func__, alignment);
        fprintf(stderr, "%s: n_tensors:    %d\n", __func__, n_tensors);
        fprintf(stderr, "%s: n_kv:         %d\n", __func__, n_kv);
        fprintf(stderr, "%s: ftype:        %s\n", __func__, ftype_str.c_str());
        fprintf(stderr, "\n");
    }
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);

    // create model object
    bert_ctx * new_bert = new bert_ctx;
    bert_model & model = new_bert->model;
    bert_vocab & vocab = new_bert->vocab;
    bert_hparams & hparams = model.hparams;

    // load hparams
    {
        hparams.n_vocab = get_u32(ctx_gguf, "vocab_size");
        hparams.n_max_tokens = get_u32(ctx_gguf, "max_position_embedding");
        hparams.n_hidden = get_u32(ctx_gguf, "hidden_size");
        hparams.n_embd = get_u32(ctx_gguf, "embedding_size");
        hparams.n_intermediate = get_u32(ctx_gguf, "intermediate_size");
        hparams.n_head = get_u32(ctx_gguf, "num_attention_heads");
        hparams.n_layer = get_u32(ctx_gguf, "num_hidden_layers");
        hparams.layer_norm_eps = get_f32(ctx_gguf, "layer_norm_eps");

        if (verbosity >= 1) {
            fprintf(stderr, "%s: MODEL\n", __func__);
            fprintf(stderr, "%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
            fprintf(stderr, "%s: n_max_tokens   = %d\n", __func__, hparams.n_max_tokens);
            fprintf(stderr, "%s: n_embd         = %d\n", __func__, hparams.n_embd);
            fprintf(stderr, "%s: n_hidden       = %d\n", __func__, hparams.n_hidden);
            fprintf(stderr, "%s: n_intermediate = %d\n", __func__, hparams.n_intermediate);
            fprintf(stderr, "%s: n_head         = %d\n", __func__, hparams.n_head);
            fprintf(stderr, "%s: n_layer        = %d\n", __func__, hparams.n_layer);
            fprintf(stderr, "%s: layer_norm_eps = %g\n", __func__, hparams.layer_norm_eps);
            fprintf(stderr, "\n");
        }
    }

    // load vocab
    {
        vocab.pad_id = get_i32(ctx_gguf, KEY_PAD_ID);
        vocab.unk_id = get_i32(ctx_gguf, KEY_UNK_ID);
        vocab.bos_id = get_i32(ctx_gguf, KEY_BOS_ID);
        vocab.eos_id = get_i32(ctx_gguf, KEY_EOS_ID);

        vocab.word_prefix = get_str(ctx_gguf, KEY_WORD_PREFIX);
        vocab.subword_prefix = get_str(ctx_gguf, KEY_SUBWORD_PREFIX);
        uint32_t word_prefix_len = vocab.word_prefix.size();
        uint32_t subword_prefix_len = vocab.subword_prefix.size();

        const int token_idx = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
        const int n_vocab = gguf_get_arr_n(ctx_gguf, token_idx);

        for (int i = 0; i < n_vocab; i++) {
            std::string word = gguf_get_arr_str(ctx_gguf, token_idx, i);
            vocab.tokens.push_back(word);

            bool subword = (
                (subword_prefix_len > 0 && word.find(vocab.subword_prefix) == 0) ||
                (word_prefix_len > 0 && word.find(vocab.word_prefix) != 0)
            );

            if (subword) {
                vocab.subword_token_to_id[word.substr(subword_prefix_len)] = i;
                vocab._id_to_subword_token[i] = word;
            } else {
                vocab.token_to_id[word.substr(word_prefix_len)] = i;
                vocab._id_to_token[i] = word;
            }
        }

        if (verbosity >= 1) {
            fprintf(stderr, "%s: TOKENIZER\n", __func__);
            fprintf(stderr, "%s: vocab size: %d\n", __func__, n_vocab);
            fprintf(stderr, "%s: word_prefix: %s\n", __func__, vocab.word_prefix.c_str());
            fprintf(stderr, "%s: subword_prefix: %s\n", __func__, vocab.subword_prefix.c_str());
            fprintf(stderr, "%s: pad_id = %d\n", __func__, vocab.pad_id);
            fprintf(stderr, "%s: unk_id = %d\n", __func__, vocab.unk_id);
            fprintf(stderr, "%s: bos_id = %d\n", __func__, vocab.bos_id);
            fprintf(stderr, "%s: eos_id = %d\n", __func__, vocab.eos_id);
            fprintf(stderr, "\n");
        }
    }

    // model tensor sizing
    size_t buffer_size = 32*1024; // need some extra room??
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            const size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
            struct ggml_tensor * cur = ggml_get_tensor(ctx_ggml, name);
            size_t tensor_size = ggml_nbytes(cur);
            buffer_size += tensor_size;
            if (verbosity >= 2) {
                fprintf(stderr, "%s: tensor[%d]: type = %s, n_dims = %d, name = %s, offset=%zu, type=%d\n", __func__, i,
                       ggml_type_name(cur->type), ggml_n_dims(cur), cur->name, offset, cur->type);
            }
        }
    }

    // initialize advanced backend
#ifdef GGML_USE_CUBLAS
    if (!use_cpu) {
        new_bert->backend = ggml_backend_cuda_init(0);
        if (!new_bert->backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        } else {
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (!use_cpu) {
        new_bert->backend = ggml_backend_metal_init();
        if (!new_bert->backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        } else {
            fprintf(stderr, "%s: using Metal backend\n", __func__);
        }
    }
#endif

    // fall back to CPU backend
    if (!new_bert->backend) {
        new_bert->backend = ggml_backend_cpu_init();
        fprintf(stderr, "%s: using CPU backend\n", __func__);
    }

    // load tensors
    {
        // host buffer for CUDA loading
        std::vector<uint8_t> read_buf;

        // context params for tensors
        struct ggml_init_params ggml_params = {
            /*.mem_size =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        // create context for tensors
        new_bert->ctx_data = ggml_init(ggml_params);
        if (!new_bert->ctx_data) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            delete new_bert;
            return nullptr;
        }

        // open model gguf file
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            fprintf(stderr, "cannot open model file for loading tensors\n");
            delete new_bert;
            return nullptr;
        }

        // add tensors to our context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * ten = ggml_get_tensor(ctx_ggml, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_bert->ctx_data, ten);
            ggml_set_name(cur, name);
        }

        // create params buffer and allocr
        new_bert->weights_buffer = ggml_backend_alloc_buffer(new_bert->backend, buffer_size);
        ggml_allocr * alloc = ggml_allocr_new_from_buffer(new_bert->weights_buffer);

        // loop over tensors and load in
        for (int i = 0; i < n_tensors; ++i) {
            // do the actual allocation on the backend
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * cur = ggml_get_tensor(new_bert->ctx_data, name);
            ggml_allocr_alloc(alloc, cur);

            // seek to the tensor data in the file
            const size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__, name);
                bert_free(new_bert);
                return nullptr;
            }

            // read in data and copy to device if needed
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(new_bert->weights_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }

        // bye bye allocr
        ggml_allocr_free(alloc);
    }

    // use get_tensors to populate bert_model
    {
        // embeddings weights
        model.word_embeddings = get_tensor(new_bert->ctx_data, "embeddings.word_embeddings.weight");
        model.token_type_embeddings = get_tensor(new_bert->ctx_data, "embeddings.token_type_embeddings.weight");
        model.position_embeddings = get_tensor(new_bert->ctx_data, "embeddings.position_embeddings.weight");
        model.embeddings_project_w = get_tensor(new_bert->ctx_data, "embeddings_project.weight");
        model.embeddings_project_b = get_tensor(new_bert->ctx_data, "embeddings_project.bias");
        model.ln_e_w = get_tensor(new_bert->ctx_data, "embeddings.LayerNorm.weight");
        model.ln_e_b = get_tensor(new_bert->ctx_data, "embeddings.LayerNorm.bias");

        // layers
        model.layers.resize(hparams.n_layer);
        for (int i = 0; i < hparams.n_layer; ++i) {
            bert_layer & layer = model.layers[i];
            std::string pre = "encoder.layer." + std::to_string(i) + ".";

            // attention
            layer.q_w = get_tensor(new_bert->ctx_data, pre + "attention.self.query.weight");
            layer.q_b = get_tensor(new_bert->ctx_data, pre + "attention.self.query.bias");
            layer.k_w = get_tensor(new_bert->ctx_data, pre + "attention.self.key.weight");
            layer.k_b = get_tensor(new_bert->ctx_data, pre + "attention.self.key.bias");
            layer.v_w = get_tensor(new_bert->ctx_data, pre + "attention.self.value.weight");
            layer.v_b = get_tensor(new_bert->ctx_data, pre + "attention.self.value.bias");

            layer.o_w = get_tensor(new_bert->ctx_data, pre + "attention.output.dense.weight");
            layer.o_b = get_tensor(new_bert->ctx_data, pre + "attention.output.dense.bias");

            layer.ln_att_w = get_tensor(new_bert->ctx_data, pre + "attention.output.LayerNorm.weight");
            layer.ln_att_b = get_tensor(new_bert->ctx_data, pre + "attention.output.LayerNorm.bias");

            // ff
            layer.ff_i_w = get_tensor(new_bert->ctx_data, pre + "intermediate.dense.weight");
            layer.ff_i_b = get_tensor(new_bert->ctx_data, pre + "intermediate.dense.bias");

            layer.ff_o_w = get_tensor(new_bert->ctx_data, pre + "output.dense.weight");
            layer.ff_o_b = get_tensor(new_bert->ctx_data, pre + "output.dense.bias");

            layer.ln_out_w = get_tensor(new_bert->ctx_data, pre + "output.LayerNorm.weight");
            layer.ln_out_b = get_tensor(new_bert->ctx_data, pre + "output.LayerNorm.bias");
        }

        // classifier
        model.classifier_w = get_tensor(new_bert->ctx_data, "classifier.weight");
        model.classifier_b = get_tensor(new_bert->ctx_data, "classifier.bias");
    }

    // free metadata
    ggml_free(ctx_ggml);
    gguf_free(ctx_gguf);

    // return context
    return new_bert;
}

// measure and allocate comptue buffers
void bert_allocate_buffers(bert_ctx * ctx, int32_t n_max_tokens, int32_t batch_size) {
    // deallocate if already allocated
    bert_deallocate_buffers(ctx);

    // get measuring allocr for backend
    ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
    ctx->compute_alloc = ggml_allocr_new_measure_from_backend(ctx->backend);

    // construct batch and compute graph
    bert_tokens tokens(n_max_tokens);
    bert_batch batch;
    for (int i = 0; i < batch_size; ++i) {
        batch.push_back(tokens);
    }
    ggml_cgraph * gf = bert_build_graph(ctx, batch, true);

    // do computing graph measurement
    size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
    ggml_allocr_free(ctx->compute_alloc);

    // now that we know the compute size, create a buffer and allocr
    ctx->compute_buffer = ggml_backend_alloc_buffer(ctx->backend, compute_memory_buffer_size);
    ctx->compute_alloc = ggml_allocr_new_from_buffer(ctx->compute_buffer);

    if (verbosity >= 1) {
        fprintf(stderr, "%s: compute allocated memory: %.2f MB\n\n", __func__, compute_memory_buffer_size / 1024.0 / 1024.0);
    }
}

void bert_deallocate_buffers(bert_ctx * ctx) {
    if (ctx->compute_buffer) {
        ggml_backend_buffer_free(ctx->compute_buffer);
        ctx->compute_buffer = NULL;
    }
    if (ctx->compute_alloc) {
        ggml_allocr_free(ctx->compute_alloc);
        ctx->compute_alloc = NULL;
    }
}

void bert_free(bert_ctx * ctx) {
    // free compute buffers
    bert_deallocate_buffers(ctx);

    // free weights buffer
    if (ctx->weights_buffer) {
        ggml_backend_buffer_free(ctx->weights_buffer);
        ctx->weights_buffer = NULL;
    }

    // free tensor context
    if (ctx->ctx_data) {
        ggml_free(ctx->ctx_data);
        ctx->ctx_data = NULL;
    }

    // free backend
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
        ctx->backend = NULL;
    }

    delete ctx;
}

//
// model execution
//

ggml_cgraph * bert_build_graph(bert_ctx * ctx, bert_batch batch, bool normalize) {
    // vocab params
    const bert_vocab & vocab = ctx->vocab;
    const bert_token pad_id = vocab.pad_id;

    // model params
    const bert_model & model = ctx->model;
    const bert_hparams & hparams = model.hparams;

    // extract model params
    const int n_embd = hparams.n_embd;
    const int n_hidden = hparams.n_hidden;
    const int n_layer = hparams.n_layer;
    const int n_max_tokens = hparams.n_max_tokens;
    const int n_head = hparams.n_head;
    const float layer_norm_eps = hparams.layer_norm_eps;
    const int d_head = n_hidden / n_head; // E = D * H

    // get the max length of the batch
    int n_batch_size = batch.size();
    int cur_max_len = 0;
    for (uint64_t ba = 0; ba < batch.size(); ba++) {
        int n = batch[ba].size();
        if (n > cur_max_len)
            cur_max_len = n;
    }

    // check for token overflow
    if (cur_max_len > n_max_tokens) {
        fprintf(stderr, "Too many tokens, maximum is %d, got %d\n", n_max_tokens, cur_max_len);
        return nullptr;
    }

    // params for graph data
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    // initialze computational graph
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, BERT_MAX_NODES, false);

    // embeddings = word_embeddings + token_type_embeddings + position_embeddings
    struct ggml_tensor * token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
    struct ggml_tensor * token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
    struct ggml_tensor * pad_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, cur_max_len, 1, n_batch_size);
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
    struct ggml_tensor * minus_one = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1); // for attention mask
    ggml_allocr_alloc(ctx->compute_alloc, token_layer);
    ggml_allocr_alloc(ctx->compute_alloc, token_types);
    ggml_allocr_alloc(ctx->compute_alloc, pad_mask);
    ggml_allocr_alloc(ctx->compute_alloc, positions);
    ggml_allocr_alloc(ctx->compute_alloc, minus_one);

    // avoid writing input embeddings in memory measure mode
    if (!ggml_allocr_is_measure(ctx->compute_alloc)) {
        int32_t * token_layer_data = (int32_t*)malloc(ggml_nbytes(token_layer));
        int32_t * token_types_data = (int32_t*)malloc(ggml_nbytes(token_types));
        float * pad_mask_data = (float*)malloc(ggml_nbytes(pad_mask));
        int32_t * pos_data = (int32_t*)malloc(ggml_nbytes(positions));
        float m1 = -1.0f;

        for (int ba = 0; ba < n_batch_size; ba++) {
            for (int i = 0; i < cur_max_len; i++) {
                int cur_len = batch[ba].size();
                if (i < cur_len) {
                    token_layer_data[ba * cur_max_len + i] = batch[ba][i];
                    pad_mask_data[ba * cur_max_len + i] = 1.0f;
                }
                else {
                    token_layer_data[ba * cur_max_len + i] = pad_id; // padding
                    pad_mask_data[ba * cur_max_len + i] = 0.0f;
                }
                token_types_data[ba * cur_max_len + i] = 0;
                pos_data[ba * cur_max_len + i] = i;
            }
        }

        ggml_backend_tensor_set(token_layer, token_layer_data, 0, ggml_nbytes(token_layer));
        ggml_backend_tensor_set(token_types, token_types_data, 0, ggml_nbytes(token_types));
        ggml_backend_tensor_set(pad_mask, pad_mask_data, 0, ggml_nbytes(pad_mask));
        ggml_backend_tensor_set(positions, pos_data, 0, ggml_nbytes(positions));
        ggml_backend_tensor_set(minus_one, &m1, 0, sizeof(m1));

        free(token_layer_data);
        free(token_types_data);
        free(pad_mask_data);
        free(pos_data);
    }

    // outer product the padding mask to kill off outside
    struct ggml_tensor * attn_mask = ggml_mul_mat(ctx0, pad_mask, pad_mask); // [L, L, 1, B]
    attn_mask = ggml_add(ctx0, attn_mask, minus_one); // result -0
    attn_mask = ggml_scale_inplace(ctx0, attn_mask, 100000.0f); // BUG: 1e3 will cause overflow?

    // get various embedding components
    struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer); // [E, L * B]
    inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.token_type_embeddings, token_types), inpL);
    inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), inpL);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, cur_max_len, n_batch_size); // [E, L, B]
    
    // embed layer norm
    inpL = ggml_norm_inplace(ctx0, inpL, layer_norm_eps);
    inpL = ggml_add(ctx0, ggml_mul(ctx0, inpL, model.ln_e_w), model.ln_e_b); // [E, L, B]

    inpL = ggml_add(ctx0, ggml_mul_mat(ctx0, model.embeddings_project_w, inpL), model.embeddings_project_b);

    // layers
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = inpL;

        // self-attention
        {
            // extract Q
            struct ggml_tensor * Q = cur;
            Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, Q), model.layers[il].q_b); // [E, L, B]
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, cur_max_len, n_batch_size); // [D, H, L, B]
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // [D, L, H, B]

            // extract K
            struct ggml_tensor * K = cur;
            K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, K), model.layers[il].k_b); // [E, L, B]
            K = ggml_reshape_4d(ctx0, K, d_head, n_head, cur_max_len, n_batch_size); // [D, H, L, B]
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3)); // [D, L, H, B]

            // extract V
            struct ggml_tensor * V = cur;
            V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, V), model.layers[il].v_b); // [E, L, B]
            V = ggml_reshape_4d(ctx0, V, d_head, n_head, cur_max_len, n_batch_size); // [D, H, L, B]
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // [L, D, H, B]

            // scaled attention
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q); // -> [L, L, H, B]
            KQ = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrt((float)d_head));
            KQ = ggml_add(ctx0, KQ, attn_mask);
            KQ = ggml_soft_max(ctx0, KQ);

            // get weighted values
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ); // -> [D, L, H, B]
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3)); // -> [D, H, L, B]

            // copy back to input (E = D * H)
            cur = ggml_reshape_3d(ctx0, KQV, n_hidden, cur_max_len, n_batch_size); // [E, L, B]
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);

        // residual connection
        cur = ggml_add(ctx0, cur, inpL);

        // attention layer norm
        cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_att_w), model.layers[il].ln_att_b);

        // store for later
        struct ggml_tensor * att_output = cur;

        // feed forward steps
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur), model.layers[il].ff_i_b);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur), model.layers[il].ff_o_b);

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, att_output, cur);

        // output layer norm
        cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_out_w), model.layers[il].ln_out_b);

        // on to next layer
        inpL = cur;
    }

    inpL = ggml_add(ctx0, ggml_mul_mat(ctx0, model.classifier_w, inpL), model.classifier_b);

    // final output
    ggml_tensor * output = inpL;

    // build the graph
    ggml_build_forward_expand(gf, output);

    // free context
    ggml_free(ctx0);

    // return complete graph
    return gf;
}

void bert_forward_batch(bert_ctx * ctx, bert_batch batch, float * embeddings, bool normalize, int32_t n_threads) {
    // reset alloc buffer to clean the memory from previous invocations
    ggml_allocr_reset(ctx->compute_alloc);

    // build the compute graph
    ggml_cgraph * gf = bert_build_graph(ctx, batch, normalize);
    if (gf == nullptr) {
        fprintf(stderr, "%s: failed to build compute graph\n", __func__);
        return;
    }

    // allocate memory for the graph
    ggml_allocr_alloc_graph(ctx->compute_alloc, gf);

    // print timing information per ggml operation (for debugging purposes)
    if (verbosity >= 3) {
        ggml_graph_print(gf);
    }

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(ctx->backend)) {
        ggml_backend_metal_set_n_cb(ctx->backend, n_threads);
    }
#endif

    // execute the graph
    ggml_backend_graph_compute(ctx->backend, gf);

    // the last node is the embedding tensor
    struct ggml_tensor * output = gf->nodes[gf->n_nodes - 1];

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(output, embeddings, 0, ggml_nbytes(output));
}

void bert_cut_batch(struct bert_ctx * ctx, bert_strings texts, float * logits, bool normalize, int32_t n_threads) {
    int32_t N = bert_n_max_tokens(ctx);
    int32_t n_input = texts.size();

    bert_batch batch;
    for (int i = 0; i < n_input; i++) {
        bert_tokens tokens = bert_tokenize(ctx, texts[i], N);
        batch.push_back(tokens);
    }

    bert_forward_batch(ctx, batch, logits, normalize, n_threads);
}

void bert_cut_batch_c(struct bert_ctx * ctx, const char ** texts, float * logits, int32_t n_input, bool normalize, int32_t n_threads) {
    bert_strings strings;
    for (int i = 0; i < n_input; i++) {
        strings.push_back(texts[i]);
    }
    bert_cut_batch(ctx, strings, logits, normalize, n_threads);
}

void bert_forward(struct bert_ctx * ctx, bert_tokens tokens, float * logits, bool normalize, int32_t n_threads) {
    bert_batch batch = {tokens};
    bert_forward_batch(ctx, batch, logits, normalize, n_threads);
}

void bert_cut(struct bert_ctx * ctx, bert_string text, float * logits, bool normalize, int32_t n_threads) {
    bert_strings strings = {text};
    bert_cut_batch(ctx, strings, logits, normalize, n_threads);
}
