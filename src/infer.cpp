#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

static void matmul(float* xout, float* x, float* w, int n, int d) {
	// W (d,n) @ x (n,) -> xout (d,)
	int i;
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
}

static void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
  float rms = 0.0f;
  for (int i = 0; i < size; ++i) {
    rms += x[i] * x[i];
  }
  rms = sqrtf(rms / size + eps);
  float scale = 1.0f / rms;
  for (int i = 0; i < size; ++i) {
    o[i] = x[i] * scale * weight[i];
  }
}

[[maybe_unused]] static void layernorm(float* o, float* x, float* weight, float* bias, int size, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < size; ++i) {
    mean += x[i];
  }
  mean /= size;
  float var = 0.0f;
  for (int i = 0; i < size; ++i) {
    var += (x[i] - mean) * (x[i] - mean);
  }
  var /= size;
  float scale = 1.0f / sqrtf(var + eps);
  if (bias) {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }
  } else {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i];
    }
  }
}

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
static void softmax(float* o, float* x, int size) {
  float score_max = -FLT_MAX;
  for (int i = 0; i < size; ++i) {
    if (x[i] > score_max) {
      score_max = x[i];
    }
  }
  float score_sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    o[i] = expf(x[i] - score_max);
    score_sum += o[i];
  }
  for (int i = 0; i < size; ++i) {
    o[i] /= score_sum;
  }
}

inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
	return x < -v ? -v : (x > v ? v : x);
}

// TODO annotate me
static void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
	for (int i = 0; i < d; i += 2) {
		int j_head = i % head_dim;
		float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
		float val = pos * freq;
		float fcr = cosf(val);
		float fci = sinf(val);

		float v0 = vec[i];
		float v1 = vec[i + 1];
		vec[i] = v0 * fcr - v1 * fci;
		vec[i + 1] = v0 * fci + v1 * fcr;
	}
}

// Compute next value in a sequence for a single causal self-attention head.
static void attn(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  float* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  float* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int kv_stride = n_kv_heads * head_dim; // stride per token in this kv head
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += qh[i] * kh[t * kv_stride + i];
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax(atth, atth, kv_len);

  // mix values with attention weights
  for (int i = 0; i < head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * vh[t * kv_stride + i];
    }
    xout[i] = vi;
  }
}

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS: 
// - `s.x()` contains the input to the block. Output will also go here.
// - The model weights are FP32.
// - Block KV cache is hydrated.
static void block(
  InferenceState& s,  // inference state
  const Config& c,    // model configuration
  Block& b,           // block weights
  int pos,            // index of the current token in the sequence
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) {
  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), b.rms_att_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  // qkv matmuls for this position
  matmul(s.q(), s.xb(), (float*)b.wq(), c.dim, q_dim);
  matmul(s.k(), s.xb(), (float*)b.wk(), c.dim, kv_dim);
  matmul(s.v(), s.xb(), (float*)b.wv(), c.dim, kv_dim);

  // some models require clipping qkv values
  for (int i = 0; i < q_dim; ++i) {
    s.q()[i] = clip(s.q()[i], c.qkv_clip);
  }
  for (int i = 0; i < kv_dim; ++i) {
    s.k()[i] = clip(s.k()[i], c.qkv_clip);
    s.v()[i] = clip(s.v()[i], c.qkv_clip);
  }

  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  rope(s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  rope(s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  
  // key and value point to the kv cache
  float* kb = b.key_cache();
  float* vb = b.value_cache();
  // update kv cache
  for (int i = 0; i < kv_dim; ++i) {
    kb[kv_pos * kv_dim + i] = s.k()[i];
    vb[kv_pos * kv_dim + i] = s.v()[i];
  }

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = c.n_heads / c.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
  for (h = 0; h < c.n_heads; ++h) {
    int head_offset = h * c.head_dim;
    float* qh = s.q() + head_offset;
    int kv_head_offset = (h / q_per_kv_head) * c.head_dim;
    float* kh = kb + kv_head_offset;
    float* vh = vb + kv_head_offset;
    attn(s.xb2() + head_offset, s.att(), qh, kh, vh, c.head_dim, c.n_kv_heads, kv_len);
  }

  // final matmul to get output of the attention, using `hb` as temp storage
  matmul(s.hb(), s.xb2(), (float*)b.wo(), q_dim, c.dim);

  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), b.rms_ffn_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul(s.hb(), s.xb(), (float*)b.w1(), c.dim, c.hidden_dim);
  matmul(s.hb2(), s.xb(), (float*)b.w3(), c.dim, c.hidden_dim);
  switch (c.act) {
    case ActivationType::GELU: {
      for (int i = 0; i < c.hidden_dim; ++i) {
        s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < c.hidden_dim; ++i) {
        s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
      }
      break;
    }
  }

  matmul(s.xb2(), s.hb(), (float*)b.w2(), c.hidden_dim, c.dim);
  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.xb2()[i];
  }
}

void forward(InferenceState& s, Model& m, int token, int pos) {
  // TODO: support float16
  assert(m.config.weight_dtype == DType::dt_f32);
  const Config& c = m.config;

  // copy the token embedding into `x`
  float* token_embedding_table = (float*)m.token_embedding_table;
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] = token_embedding_table[token * c.dim + i];
  }

  // TODO: attention sinks
	int kv_pos = pos % c.max_seq_len;
	int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  // forward all layers in order
  for (auto& b : m.blocks) {
    block(s, c, b, pos, kv_pos, kv_len);
  }

  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.x(), s.x(), m.rms_final_weight, c.dim, c.norm_eps);
      break;
    }
  }

  // classifier into logits
  matmul(s.logits(), s.x(), (float*)m.wcls, c.dim, c.vocab_size);
}