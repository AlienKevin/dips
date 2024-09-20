#include "bert.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <vector>

bert_ctx * g_context = nullptr;

EMSCRIPTEN_BINDINGS(bert) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        if (g_context == nullptr) {
            g_context = bert_load_from_file(path_model.c_str(), false);
            if (g_context != nullptr) {
                // allocate buffer for building compute graph
                int batch_size = 1;
                bert_allocate_buffers(g_context, bert_n_max_tokens(g_context), batch_size);
                return true;
            } else {
                return false;
            }
        }

        return false;
    }));

    emscripten::function("free", emscripten::optional_override([]() {
        if (g_context != nullptr) {
            bert_free(g_context);
            g_context = nullptr;
        }
    }));

    emscripten::function("run", emscripten::optional_override([](const std::string & input) {
        if (g_context == nullptr) {
            return emscripten::val::null();
        }
        bert_ctx * ctx = g_context;
        int n_max_tokens = bert_n_max_tokens(ctx);
        bert_tokens tokens = bert_tokenize(ctx, input.c_str(), n_max_tokens);

        std::vector<float> logits(tokens.size() * 4);  // 4 logits per token
        bert_batch batch = { tokens };
        bert_forward_batch(ctx, batch, logits.data(), true, 1);

        return emscripten::val(emscripten::typed_memory_view(logits.size(), logits.data()));
    }));
}
