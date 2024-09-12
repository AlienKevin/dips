#ifndef BERT_H
#define BERT_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

// model keys

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"

#define KEY_PAD_ID "tokenizer.ggml.padding_token_id"
#define KEY_UNK_ID "tokenizer.ggml.unknown_token_id"
#define KEY_BOS_ID "tokenizer.ggml.bos_token_id"
#define KEY_EOS_ID "tokenizer.ggml.eos_token_id"
#define KEY_WORD_PREFIX "tokenizer.ggml.word_prefix"
#define KEY_SUBWORD_PREFIX "tokenizer.ggml.subword_prefix"
#define KEY_TOKEN_LIST "tokenizer.ggml.tokens"

// api

#define BERT_API __attribute__ ((visibility ("default")))

#ifdef __cplusplus
extern "C" {
#endif

//
// data types
//

typedef int32_t bert_token;
typedef std::vector<bert_token> bert_tokens;
typedef std::vector<bert_tokens> bert_batch;
typedef std::string bert_string;
typedef std::vector<bert_string> bert_strings;

//
// data structures
//

// model params
struct bert_hparams {
    int32_t n_vocab;
    int32_t n_max_tokens;
    int32_t n_embd;
    int32_t n_hidden;
    int32_t n_intermediate;
    int32_t n_head;
    int32_t n_layer;
    float_t layer_norm_eps;
};

struct bert_layer {
    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;
};

struct bert_vocab {
    bert_token pad_id;
    bert_token unk_id;
    bert_token bos_id;
    bert_token eos_id;

    std::string word_prefix;
    std::string subword_prefix;

    std::vector<std::string> tokens;

    std::map<std::string, bert_token> token_to_id;
    std::map<std::string, bert_token> subword_token_to_id;

    std::map<bert_token, std::string> _id_to_token;
    std::map<bert_token, std::string> _id_to_subword_token;
};

struct bert_model {
    bert_hparams hparams;

    // embeddings weights
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;
    struct ggml_tensor *embeddings_project_w;
    struct ggml_tensor *embeddings_project_b;
    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<bert_layer> layers;

    struct ggml_tensor *classifier_w;
    struct ggml_tensor *classifier_b;
};

struct bert_ctx {
    bert_model model;
    bert_vocab vocab;

    // ggml context
    struct ggml_context * ctx_data = NULL;

    // compute metadata
    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t weights_buffer = NULL;
    ggml_backend_buffer_t compute_buffer = NULL;
    ggml_allocr * compute_alloc = NULL;
};

//
// main api
//

BERT_API struct bert_ctx * bert_load_from_file(
    const char * fname,
    bool use_cpu
);

BERT_API void bert_allocate_buffers(
    bert_ctx * ctx,
    int32_t n_max_tokens,
    int32_t batch_size
);

BERT_API void bert_deallocate_buffers(bert_ctx * ctx);
BERT_API void bert_free(bert_ctx * ctx);

BERT_API ggml_cgraph * bert_build_graph(
    bert_ctx * ctx,
    bert_batch batch,
    bool normalize
);

BERT_API void bert_forward_batch(
    bert_ctx * ctx,
    bert_batch tokens,
    float * embeddings,
    bool normalize,
    int32_t n_thread
);

BERT_API void bert_cut_batch(
    struct bert_ctx * ctx,
    bert_strings texts,
    float * logits,
    bool normalize,
    int32_t n_threads
);

BERT_API void bert_cut_batch_c(
    struct bert_ctx * ctx,
    const char ** texts,
    float * logits,
    int32_t n_input,
    bool normalize,
    int32_t n_threads
);

BERT_API bert_tokens bert_tokenize(
    struct bert_ctx * ctx,
    bert_string text,
    uint64_t n_max_tokens
);

BERT_API bert_string bert_detokenize(
    struct bert_ctx * ctx,
    bert_tokens tokens,
    bool debug
);

BERT_API uint64_t bert_tokenize_c(
    struct bert_ctx * ctx,
    const char * text,
    int32_t * output,
    uint64_t n_max_tokens
);

BERT_API uint64_t bert_detokenize_c(
    struct bert_ctx * ctx,
    int32_t * tokens,
    char * output,
    uint64_t n_tokens,
    uint64_t n_output,
    bool debug
);

BERT_API void bert_forward(
    struct bert_ctx * ctx,
    bert_tokens tokens,
    float * embeddings,
    bool normalize,
    int32_t n_thread
);

BERT_API void bert_cut(
    struct bert_ctx * ctx,
    bert_string text,
    float * logits,
    bool normalize,
    int32_t n_threads
);

BERT_API int32_t bert_n_embd(bert_ctx * ctx);
BERT_API int32_t bert_n_max_tokens(bert_ctx * ctx);

BERT_API const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_token id);

#ifdef __cplusplus
}
#endif

#endif // BERT_H
