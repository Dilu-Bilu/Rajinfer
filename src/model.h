#pragma once

#include "codec.h"

#include <memory>
#include <vector>

enum class ActivationType {
  GELU,
  SILU,
};

enum class LayerNormType {
  RMSNorm,
};

struct Config {
  int dim;                  // transformer input & output dimension
  int hidden_dim;           // dimension of hidden layer in feedforward network
  int head_dim;             // dimension of each attention head, usually dim / n_heads
  int n_layers;             // number of layers
  int n_heads;              // number of attention query heads
  int n_kv_heads;           // number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int vocab_size;           // vocabulary size
  int max_seq_len;          // max sequence length
  float rope_theta;         // RoPE theta
  int rotary_dim;           // dimension of rotary position encoding (elements after that don't get rotated)
  float norm_eps;           // epsilon for layer normalization
  ActivationType act;       // activation function
  LayerNormType norm_type;  // norm type
  float qkv_clip;           // clip qkv values to [-clip, clip]

  // Data type of the weights according to config, used
  // to safety check tensor dtype at initialization time.
  DType weight_dtype;

  // If nonzero `context` is supplied, max sequence length is limited to `context`.
  void from_yalm(YALMData& yalm, int context = 0);
  size_t active_bytes(size_t pos) const;
};

struct Block {
  /* Transformer Block */

  Block(
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
  );

  float* rms_att_weight() const { return _rms_att_weight; }
  float* rms_ffn_weight() const { return _rms_ffn_weight; }
  void* wq() const { return _wq; }
  void* wk() const { return _wk; }
  void* wv() const { return _wv; }
  void* wo() const { return _wo; }
  void* w1() const { return _w1; }
  void* w2() const { return _w2; }
  void* w3() const { return _w3; }
  float* key_cache() const { return _key_cache.get(); }
  float* value_cache() const { return _value_cache.get(); }

private:
  // weights for norms
	float* _rms_att_weight = nullptr; // (dim) rmsnorm weights
	float* _rms_ffn_weight = nullptr; // (dim)

  // weights for self-attention matmuls
	void* _wq = nullptr; // (n_heads * head_dim, dim)
	void* _wk = nullptr; // (n_kv_heads * head_dim, dim)
	void* _wv = nullptr; // (n_kv_heads * head_dim, dim)
	void* _wo = nullptr; // (dim, n_heads * head_dim)
	
  // weights for ffn
	void* _w1 = nullptr; // (n_experts?, hidden_dim, dim)
	void* _w2 = nullptr; // (n_experts?, dim, hidden_dim)
	void* _w3 = nullptr; // (n_experts?, hidden_dim, dim) - GLU weights

  // kv cache
	std::shared_ptr<float[]> _key_cache = nullptr;   // (seq_len, n_kv_heads * head_dim)
	std::shared_ptr<float[]> _value_cache = nullptr; // (seq_len, n_kv_heads * head_dim)
};

// Buffer for all state used during a forward pass.
// Members are reused across subsequent blocks and passes.
// This lets us avoid allocations during inference.
struct InferenceState {
  InferenceState(const Config& config);

  // current activations
  float* x() const { return _x.get(); }
  float* xb() const { return _xb.get(); }
  // TODO: do we need xb2?
  float* xb2() const { return _xb2.get(); }
  float* hb() const { return _hb.get(); }
  float* hb2() const { return _hb2.get(); }
  float* q() const { return _q.get(); }
  float* k() const { return _k.get(); }
  float* v() const { return _v.get(); }
  float* att() const { return _att.get(); }
  // LM head
  float* logits() const { return _logits.get(); }

private:
  // current activations
  std::unique_ptr<float[]> _x = nullptr;         // (dim,) - latest activation
  std::unique_ptr<float[]> _xb = nullptr;        // (dim,) - activation inside a residual branch
  // TODO: do we need xb2?
  std::unique_ptr<float[]> _xb2 = nullptr;       // (dim,) - activation inside a residual branch (second slot)
  std::unique_ptr<float[]> _hb = nullptr;        // (hidden_dim,) - buffer for hidden dimension in feedforward network
  std::unique_ptr<float[]> _hb2 = nullptr;       // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
  std::unique_ptr<float[]> _q = nullptr;         // (n_heads * head_dim,) - query vectors for latest timestamp
  std::unique_ptr<float[]> _k = nullptr;         // (n_kv_heads * head_dim,) - key vectors for latest timestamp
  std::unique_ptr<float[]> _v = nullptr;         // (n_kv_heads * head_dim,) - value vectors for latest timestamp
  std::unique_ptr<float[]> _att = nullptr;       // (n_heads, seq_len) - buffer for attention scores
  // LM head
  std::unique_ptr<float[]> _logits = nullptr;    // (vocab_size,) - final output logits
};

struct Model {
  Config config;

  std::vector<Block> blocks;
  
  // token embedding table
	void* token_embedding_table = nullptr; // (vocab_size, dim)
  // final norm
	float* rms_final_weight = nullptr; // (dim,)
	// classifier weights for the logits, on the last layer
	void* wcls = nullptr; // (vocab_size, dim)

  Model(YALMData& yalm);
};

void forward(InferenceState& s, Model& m, int token, int pos);