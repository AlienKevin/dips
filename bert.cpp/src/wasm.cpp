#include "bert.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <vector>

std::vector<struct bert_ctx *> g_contexts(4, nullptr);

EMSCRIPTEN_BINDINGS(bert) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = bert_load_from_file(path_model.c_str(), false);
                if (g_contexts[i] != nullptr) {
                    // allocate buffer for building compute graph
                    int batch_size = 1;
                    bert_allocate_buffers(g_contexts[i], bert_n_max_tokens(g_contexts[i]), batch_size);
                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        --index;

        if (index < g_contexts.size()) {
            bert_free(g_contexts[index]);
            g_contexts[index] = nullptr;
        }
    }));

    emscripten::function("run", emscripten::optional_override([](size_t index, const std::string & input) {
        --index;

        if (index >= g_contexts.size() || g_contexts[index] == nullptr) {
            return emscripten::val::null();
        }

        bert_ctx * ctx = g_contexts[index];
        int n_max_tokens = bert_n_max_tokens(ctx);
        bert_tokens tokens = bert_tokenize(ctx, input.c_str(), n_max_tokens);

        std::vector<float> logits(tokens.size() * 4);  // 4 logits per token
        bert_batch batch = { tokens };
        bert_forward_batch(ctx, batch, logits.data(), true, 1);

        return emscripten::val(emscripten::typed_memory_view(logits.size(), logits.data()));
    }));
}
