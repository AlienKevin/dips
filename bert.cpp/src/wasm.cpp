#include "bert.h"
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <vector>
#include <string>

// Global context
static bert_ctx* g_context = nullptr;

// Core implementation functions
bool init_impl(const std::string& path_model) {
    if (g_context == nullptr) {
        g_context = bert_load_from_file(path_model.c_str(), false);
        if (g_context != nullptr) {
            int batch_size = 1;
            bert_allocate_buffers(g_context, bert_n_max_tokens(g_context), batch_size);
            return true;
        }
    }
    return false;
}

void free_impl() {
    if (g_context != nullptr) {
        bert_free(g_context);
        g_context = nullptr;
    }
}

std::vector<float> run_impl(const std::string& input) {
    if (g_context == nullptr) {
        return {};
    }
    bert_ctx* ctx = g_context;
    int n_max_tokens = bert_n_max_tokens(ctx);
    bert_tokens tokens = bert_tokenize(ctx, input.c_str(), n_max_tokens);

    std::vector<float> logits(tokens.size() * 4);  // 4 logits per token
    bert_batch batch = { tokens };
    bert_forward_batch(ctx, batch, logits.data(), true, 1);

    return logits;
}

// EMSCRIPTEN_KEEPALIVE functions
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    bool init(const std::string& path_model) {
        return init_impl(path_model);
    }

    EMSCRIPTEN_KEEPALIVE
    void free_ctx() {
        free_impl();
    }

    EMSCRIPTEN_KEEPALIVE
    float* run(const std::string& input, int* output_size) {
        std::vector<float> result = run_impl(input);
        *output_size = result.size();
        float* output = (float*)malloc(result.size() * sizeof(float));
        memcpy(output, result.data(), result.size() * sizeof(float));
        return output;
    }
}

// EMSCRIPTEN_BINDINGS
EMSCRIPTEN_BINDINGS(bert) {
    emscripten::function("init", &init_impl);
    emscripten::function("free", &free_impl);
    emscripten::function("run", emscripten::optional_override([](const std::string& input) {
        std::vector<float> result = run_impl(input);
        return emscripten::val(emscripten::typed_memory_view(result.size(), result.data()));
    }));
}
