#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "fmt/format.h"

#include "codec.h"
#include "model.h"
#include "sampler.h"
#include "time.h"
#include "tokenizer.h"


void error_usage() {
	fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
	fprintf(stderr, "Example: run model.yalm -i \"Q: What is the meaning of life?\"\n");
	fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -m [completion,perplexity] which mode to run in (default - completion)\n");
  fprintf(stderr, "  Choose one:\n");
	fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Completion mode options:\n");
  fprintf(stderr, "  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite\n");
	exit(1);
}

int main(int argc, char* argv[]) {
  std::string checkpoint_path = "";    // e.g. out/model.bin
  // Options
  std::string mode = "completion";     // completion or perplexity
  std::string prompt = "";             // prompt string
  std::string prompt_path = "";        // prompt file path
  // Completion mode options
  int num_steps = 256;                 // number of steps to run for

	if (argc >= 2) {
		checkpoint_path = argv[1];
	} else {
		error_usage();
	}
	for (int i = 2; i < argc;) {
		// do some basic validation
		if (i + 1 >= argc) {
			error_usage();
		} // must have arg after flag
		if (argv[i][0] != '-') {
			error_usage();
		} // must start with dash
		if (strlen(argv[i]) != 2) {
			error_usage();
		} // must be -x (one dash, one letter)

		// read in the args
		if (argv[i][1] == 'm') {
      if (i + 1 >= argc) {
        error_usage();
      }
      mode = argv[i + 1];
      if (std::string("completion").starts_with(mode)) {
        mode = "completion";
      } else if (std::string("perplexity").starts_with(mode)) {
        mode = "perplexity";
      } else {
        error_usage();
      }
      i += 2;
    } else if (argv[i][1] == 'i') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt = argv[i + 1];
      i += 2;
		} else if (argv[i][1] == 'f') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt_path = argv[i + 1];
      i += 2;
    } else if (argv[i][1] == 'n') {
      if (i + 1 >= argc) {
        error_usage();
      }
      num_steps = std::stoi(argv[i + 1]);
      i += 2;
    } else {
			error_usage();
		}
	}
  int has_prompt = prompt.size() > 0 ? 1 : 0;
  int has_prompt_path = prompt_path.size() > 0 ? 1 : 0;
  if ((has_prompt + has_prompt_path) != 1) {
    error_usage();
  } else if (has_prompt_path) {
    std::ifstream file(prompt_path);
    if (!file.is_open()) {
      std::cerr << "Error: could not open file " << prompt_path << std::endl;
      return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    prompt = buffer.str();
  }

  YALMData model_data;
  model_data.from_file(checkpoint_path);
  Model model(model_data);
  InferenceState state(model.config);
  Sampler sampler(model.config);
  Tokenizer tokenizer(model_data);

  if (num_steps == 0) {
    // `-n 0` means use the full context length
    num_steps = model.config.max_seq_len;
  }

  // Do one inference as warmup.
  // On CPU, this ensures all tensors are loaded into memory via mmap.
  // On GPU, this ensures all tensors are loaded into device memory and 
  // kernels are compiled + instantiated.
  forward(state, model, 0, 0);

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
    uint64_t encoding_ms = encode_end_ms - encode_start_ms;
    std::cout << fmt::format(
      "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
      encoding.size(),
      encoding.size() / (encoding_ms / 1000.0),
      (encoding_ms / 1000.0) / encoding.size(),
      encoding_ms / 1000.0
    ) << std::endl;
  }

  if (mode == "completion") {
    uint64_t start_ms = get_timestamp_ms();
    size_t read_bytes = 0;
    // Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
    // This also generates output logits for the last token.
    for (size_t pos = 0; pos < encoding.size(); pos++) {
      int token_id = encoding[pos];
      forward(state, model, token_id, pos);
      read_bytes += model.config.active_bytes(pos);
    }
    uint64_t end_hydrate_ms = get_timestamp_ms();
    // For N steps:
    // - Sample + decode output logits
    // - Forward the model
    for (int i = 0; i < num_steps || num_steps == -1; i++) {
      int token_id = sampler.sample_argmax(state);
      std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
      std::cout << token_str << std::flush;
      encoding.push_back(token_id);
      if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
        break;
      }
      forward(state, model, token_id, encoding.size() - 1);
      read_bytes += model.config.active_bytes(encoding.size() - 1);
    }
    std::cout << "\n" << std::endl;
    uint64_t end_ms = get_timestamp_ms();
    double elapsed_s = (end_ms - start_ms) / 1000.0;
    std::cout << fmt::format(
      "Generation stats:\n"
      "  {} tokens\n"
      "  throughput: {:.5}tok/s\n"
      "  latency: {:.5}s/tok\n"
      "  hydrate: {:.5}s\n"
      "  bandwidth: {:.5}GB/s\n"
      "  total: {:.5}s\n",
      encoding.size(),
      encoding.size() / elapsed_s,
      elapsed_s / encoding.size(),
      (end_hydrate_ms - start_ms) / 1000.0,
      ((double)read_bytes / 1e9) / elapsed_s,
      elapsed_s
    ) << std::endl;
  } else {
    double sum_logprob = 0.0;
    double ss_logprob = 0.0;
    // Generates output logits for all tokens in the prompt and sum log probs to
    // compute perplexity.
    uint64_t start_ms = get_timestamp_ms();
    size_t read_bytes = 0;
    size_t N = encoding.size() - 1;
    for (size_t pos = 0; pos + 1 < encoding.size(); pos++) {
      std::cout << "\r Computing perplexity..." << pos + 1 << "/" << N << std::flush;
      
      int token_id = encoding[pos];
      forward(state, model, token_id, pos);
      read_bytes += model.config.active_bytes(pos);

      double logprob = std::log(sampler.sample_prob(encoding[pos + 1], state));
      sum_logprob += logprob;
      ss_logprob += logprob * logprob;
    }
    std::cout << std::endl;
    uint64_t end_ms = get_timestamp_ms();
    double elapsed_s = (end_ms - start_ms)/1000.0;
    double perplexity = std::exp(-sum_logprob / N);
    double perplexity_error = perplexity * std::sqrt(
      (ss_logprob - sum_logprob * sum_logprob / N) / N / N
    );
    std::cout << fmt::format(
      "Stats:\n"
      "  {} tokens\n"
      "  perplexity: {:.5} ± {:.5}\n"
      "  throughput: {:.5}tok/s\n"
      "  latency: {:.5}s/tok\n"
      "  bandwidth: {:.5}GB/s\n"
      "  total: {:.5}s\n",
      N,
      perplexity,
      perplexity_error,
      N / elapsed_s,
      elapsed_s / N,
      ((double)read_bytes / 1e9) / elapsed_s,
      elapsed_s
    ) << std::endl;
  }

  return 0;
}