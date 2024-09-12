#include "bert.h"
#include "ggml.h"

#include <unistd.h>
#include <stdio.h>
#include <vector>

struct bert_options
{
    const char* model = nullptr;
    const char* prompt = nullptr;
    int32_t n_max_tokens = 0;
    int32_t batch_size = 32;
    bool use_cpu = false;
    bool normalize = true;
    int32_t n_threads = 6;
};

void bert_print_usage(char **argv, const bert_options &options) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -r --raw              don't normalize embeddings (default: normalize embeddings)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", options.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model GGUF path\n");
    fprintf(stderr, "  -n N, --max-tokens N  number of tokens to generate (default: max)\n");
    fprintf(stderr, "  -b BATCH_SIZE, --batch-size BATCH_SIZE\n");
    fprintf(stderr, "                        batch size to use when executing model\n");
    fprintf(stderr, "  -c, --cpu             use CPU backend (default: use CUDA if available)\n");
    fprintf(stderr, "\n");
}

bool bert_options_parse(int argc, char **argv, bert_options &options) {
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-r" || arg == "--raw") {
            options.normalize = false;
        } else if (arg == "-t" || arg == "--threads") {
            options.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            options.prompt = argv[++i];
        } else if (arg == "-m" || arg == "--model") {
            options.model = argv[++i];
        } else if (arg == "-n" || arg == "--max-tokens") {
            options.n_max_tokens = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--cpu") {
            options.use_cpu = true;
        } else if (arg == "-h" || arg == "--help") {
            bert_print_usage(argv, options);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, options);
            exit(0);
        }
    }

    if (options.model == nullptr) {
        fprintf(stderr, "error: model file is required\n");
        bert_print_usage(argv, options);
        return false;
    }

    if (options.prompt == nullptr) {
        fprintf(stderr, "error: prompt is required\n");
        bert_print_usage(argv, options);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    // load cli arguments
    bert_options options;
    if (bert_options_parse(argc, argv, options) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    bert_ctx * bctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        bctx = bert_load_from_file(options.model, options.use_cpu);
        if (bctx == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, options.model);
            return 1;
        }

        // allocate buffers for length and batch size
        if (options.n_max_tokens <= 0) {
            options.n_max_tokens = bert_n_max_tokens(bctx);
        }
        bert_allocate_buffers(bctx, options.n_max_tokens, options.batch_size);

        t_load_us = ggml_time_us() - t_start_us;
    }

    int64_t t_start_us = ggml_time_us();

    // tokenize the prompt
    int N = bert_n_max_tokens(bctx);
    bert_tokens tokens = bert_tokenize(bctx, options.prompt, N);

    int64_t t_mid_us = ggml_time_us();
    int64_t t_token_us = t_mid_us - t_start_us;

    // print the tokens
    for (auto & tok : tokens) {
        fprintf(stderr, "%d -> %s\n", tok, bert_vocab_id_to_token(bctx, tok));
    }
    fprintf(stderr, "\n");

    // create a batch
    const int n_embd = bert_n_embd(bctx);
    bert_batch batch = { tokens };

    // run the embedding
    std::vector<float> embed(batch.size()*n_embd);
    bert_forward_batch(bctx, batch, embed.data(), options.normalize, options.n_threads);

    int64_t t_end_us = ggml_time_us();
    int64_t t_eval_us = t_end_us - t_mid_us;

    printf("[ ");
    for (int i = 0; i < n_embd; i++) {
        const char * sep = (i == n_embd - 1) ? "" : ",";
        printf("%1.4f%s ",embed[i], sep);
    }
    printf("]\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n");
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        fprintf(stderr, "%s:    token time = %8.2f ms / %.2f ms per token\n", __func__, t_token_us/1000.0f, t_token_us/1000.0f/tokens.size());
        fprintf(stderr, "%s:     eval time = %8.2f ms / %.2f ms per token\n", __func__, t_eval_us/1000.0f, t_eval_us/1000.0f/tokens.size());
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}
