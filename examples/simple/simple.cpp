#include "common.h"
#include "llama.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static const char* prompt_template3 = "<|user|>%s<|end|><|assistant|>";

static std::string format_prompt(const char* input)
{
    size_t len = strlen(input) + strlen(prompt_template3) - 1;

    std::string text(len, 0);
    sprintf(&text[0], prompt_template3, input);
    text.pop_back();

    return text;
}

static std::string passkey_prompt(int n_junk = 500, int i_pos = 333)
{
    const std::string prompt_prefix = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.";
    const std::string prompt_suffix = " What is the pass key? The pass key is";

    // generate junk text
    auto prompt = prompt_prefix;

    const int passkey = rand() % 50000 + 1;

    for (int i = 0; i < n_junk; i++) {
        if (i % n_junk == i_pos) {
            prompt += " The pass key is " + std::to_string(passkey) + ". Remember it. " + std::to_string(passkey) + " is the pass key.";
        }

        prompt += " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.";
    }

    prompt += prompt_suffix;
    return prompt;
}

std::vector<int> read_tokens() {
    // Specify the file path
    const char* file_path = "/mnt/models/Phi-3-mini-128k-instruct/ids.txt";

    // Open the file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << file_path << std::endl;
        return {};
    }

    // Vector to store tokens
    std::vector<int> tokens;

    // Read tokens from the file, treating each line as a single integer
    std::string line;
    while (std::getline(file, line)) {
        // Convert string to integer using atoi function
        int num = std::atoi(line.c_str());
        tokens.push_back(num);
    }

    // Close the file
    file.close();

    return tokens;
}

int main(int argc, char ** argv) {
    gpt_params params;

    // if (argc == 1 || argv[1][0] == '-') {
    //     printf("usage: %s MODEL_PATH [PROMPT]\n" , argv[0]);
    //     return 1 ;
    // }

    // if (argc >= 2) {
    //     params.model = argv[1];
    // }

    // if (argc >= 3) {
    //     params.prompt = argv[2];
    // }

    // if (params.prompt.empty()) {
    //     params.prompt = passkey_prompt();
    // }

    params.model = "/mnt/models/phi3_n16.gguf";
    params.prompt = format_prompt(passkey_prompt(10).c_str());

    // total length of the sequence including the prompt
    const int n_len = 32;
    params.n_threads = 1;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    model_params.n_gpu_layers = 99; // offload all layers to the GPU
    // model_params.split_mode = LLAMA_SPLIT_MODE_NONE;

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 8192;
    ctx_params.n_ubatch = 8192;
    ctx_params.n_batch = 8192;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    // ctx_params.offload_kqv = false;
    
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list = {1, 32010, 2649, 592, 263, 2958, 446, 32007, 32001};
    // tokens_list = ::llama_tokenize(ctx, params.prompt, true, true);
    tokens_list = read_tokens();

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    // for (auto id : tokens_list) {
    //     fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    // }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(ctx_params.n_ctx, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (true) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_len) {
                LOG_TEE("\n");

                break;
            }

            LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
