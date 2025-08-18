#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <array>
#include <cfloat>
#include "fmt/format.h"
#include <iostream>
#include <limits.h>
#include <string>

using json = nlohmann::json;

void Config::from_yalm(YALMData& yalm, int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
	hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
	head_dim = std::stoi(yalm.metadata.at("head_dim").get<std::string>());
	n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
	n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
	n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
	vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());

	// for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
	max_seq_len = std::min(std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
		max_seq_len = context;
	}

	rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
	rotary_dim = std::stoi(yalm.metadata.at("rotary_dim").get<std::string>());

	norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cerr << "unsupported act_type, defaulting to gelu" << std::endl;
    act = ActivationType::GELU;
  }

	std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

	qkv_clip = yalm.metadata.contains("qkv_clip") ? std::stof(yalm.metadata.at("qkv_clip").get<std::string>()) : FLT_MAX;

  std::string dtype = yalm.metadata.at("dtype").get<std::string>();
  // TODO: support fp16
  // TODO: support fp8
  if (dtype == "fp32") {
    weight_dtype = DType::dt_f32;
  } else {
    std::cerr << "FATAL: unsupported dtype: " << dtype << std::endl;
    assert(false);
  }
}

size_t Config::active_bytes(size_t pos) const {
  size_t weight_size = dtype_size(weight_dtype);

  size_t bytes_per_block = 0;
  bytes_per_block += 2 * dim * sizeof(float); // rms_att_weight, rms_ffn_weight
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wq
  bytes_per_block += 2 * n_kv_heads * head_dim * dim * weight_size; // wk, wv
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wo
  bytes_per_block += 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  size_t kv_len = std::min(static_cast<size_t>(max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(float);
  bytes_per_block += 2 * kv_len * n_kv_heads * head_dim * kv_entry_size; // key_cache, value_cache

  size_t bytes = 0;
  bytes += dim * weight_size; // 1 row of token_embedding_table
  bytes += n_layers * bytes_per_block; // blocks
  bytes += dim * sizeof(float); // rms_final_weight
  bytes += vocab_size * dim * sizeof(float); // wcls

  return bytes;
}

void* check_tensor(const Tensor* tensor, DType weight_dtype, std::array<int, 4> shape) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor" << std::endl;
    assert(false);
    return nullptr;
  }
  if (tensor->dtype != weight_dtype || tensor->shape != shape) {
    std::cerr << "FATAL: tensor mismatch for " << tensor->name << std::endl;
    std::cerr 
      << fmt::format("expected: dtype={}, shape=[{},{},{},{}]", dtype_to_string(weight_dtype), shape[0], shape[1], shape[2], shape[3]) 
      << std::endl;
    std::cerr 
      << fmt::format("got: dtype={}, shape=[{},{},{},{}]", dtype_to_string(tensor->dtype), tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]) 
      << std::endl;
    assert(false);
  }
  return tensor->data;
};

const Tensor* get_tensor(const YALMData& yalm, const std::string& key) {
  auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor: " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor& tensor = it->second;
  return &tensor;
};

Block::Block(
  const Config& config,
  const Tensor* rms_att_weight,
  const Tensor* rms_ffn_weight,
  const Tensor* wq,
  const Tensor* wk,
  const Tensor* wv,
  const Tensor* wo,
  const Tensor* w1,
  const Tensor* w2,
  const Tensor* w3
) {

  _rms_att_weight = static_cast<float*>(check_tensor(
    rms_att_weight, DType::dt_f32, {config.dim, 0, 0, 0}
  ));
  _rms_ffn_weight = static_cast<float*>(check_tensor(
    rms_ffn_weight, DType::dt_f32, {config.dim, 0, 0, 0}
  ));

  _wq = check_tensor(
    wq, config.weight_dtype, {config.n_heads * config.head_dim, config.dim, 0, 0}
  );
  _wk = check_tensor(
    wk, config.weight_dtype, {config.n_kv_heads * config.head_dim, config.dim, 0, 0}
  );
  _wv = check_tensor(
    wv, config.weight_dtype, {config.n_kv_heads * config.head_dim, config.dim, 0, 0}
  );
  _wo = check_tensor(
    wo, config.weight_dtype, {config.dim, config.n_heads * config.head_dim, 0, 0}
  );

  _w1 = check_tensor(
    w1, config.weight_dtype, {config.hidden_dim, config.dim, 0, 0}
  );
  _w2 = check_tensor(
    w2, config.weight_dtype, {config.dim, config.hidden_dim, 0, 0}
  );
  _w3 = check_tensor(
    w3, config.weight_dtype, {config.hidden_dim, config.dim, 0, 0}
  );

  _key_cache.reset(new float[config.max_seq_len * config.n_kv_heads * config.head_dim]());
  _value_cache.reset(new float[config.max_seq_len * config.n_kv_heads * config.head_dim]());
}

InferenceState::InferenceState(const Config& config) {
  _x.reset(new float[config.dim]());
  _xb.reset(new float[config.dim]());
  _xb2.reset(new float[config.dim]());
  _hb.reset(new float[config.hidden_dim]());
  _hb2.reset(new float[config.hidden_dim]());
  _q.reset(new float[config.n_heads * config.head_dim]());
  _k.reset(new float[config.n_kv_heads * config.head_dim]());
  _v.reset(new float[config.n_kv_heads * config.head_dim]());
  _att.reset(new float[config.n_heads * config.max_seq_len]());
  _logits.reset(new float[config.vocab_size]());
}

Model::Model(YALMData& yalm) {
  config.from_yalm(yalm);
  std::cout << "loading model with dtype: " << dtype_to_string(config.weight_dtype) << std::endl;

  token_embedding_table = check_tensor(
    get_tensor(yalm, "model.embed.weight"), 
    config.weight_dtype,
    {config.vocab_size, config.dim, 0, 0}
  );

  for (int i = 0; i < config.n_layers; ++i) {
    blocks.emplace_back(
      config,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wk.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wv.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.weight", i))
    );
  }

  rms_final_weight = static_cast<float*>(check_tensor(
    get_tensor(yalm, "model.norm.weight"), 
    DType::dt_f32, 
    {config.dim, 0, 0, 0}
  ));
  wcls = check_tensor(
    get_tensor(yalm, "model.output.weight"), 
    config.weight_dtype, 
    {config.vocab_size, config.dim, 0, 0}
  );
}