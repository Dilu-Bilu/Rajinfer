// #include "model.h"

// #include <cfloat>
// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>

// #define FULL_MASK 0xffffffff

// // This is a helper function to check for CUDA errors
// // cudaError is a type to define error codes from CUDA runtime API 
// #define CUDA_CHECK(x)                                                                                    \
// 	do {                                                                                                 \
// 		cudaError_t err = x;                                                                             \
// 		if (err != cudaSuccess) {                                                                        \
// 			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
// 			        cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
// 			abort();                                                                                     \
// 		}                                                                                                \
// 	} while (0)

// // This is a helper function to copy memory from host to device
// static void* cuda_devicecopy(void* host, size_t size) {
// 	void* device = NULL;
// 	CUDA_CHECK(cudaMalloc(&device, size));
// 	CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
// 	return device;
// }

// [[maybe_unused]] static void* cuda_devicealloc(size_t size) {
// 	void* ptr = NULL;
// 	CUDA_CHECK(cudaMalloc(&ptr, size));
// 	return ptr;
// }

// [[maybe_unused]] static void* cuda_hostalloc(size_t size) {
// 	void* ptr = NULL;
// 	CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
// 	return ptr;
// }

// // These are C-style wrappers for the above functions to be called from C code
// extern "C" void* upload_cuda(void* host, size_t size) {
// 	return cuda_devicecopy(host, size);
// }

// // Register a pinned host memory region with CUDA
// extern "C" void register_cuda_host(void* host, size_t size) {
//   CUDA_CHECK(cudaHostRegister(&host, size, cudaHostRegisterDefault));
//   CUDA_CHECK(cudaHostRegister(host, size, cudaHostRegisterDefault));
// }

// // This frees device memory allocated by cuda_devicecopy or cuda_devicealloc
// extern "C" void free_cuda(void* device) {
//   CUDA_CHECK(cudaFree(device));
// }

// // This frees host memory allocated by register host cuda 
// extern "C" void unregister_cuda_host(void* host) {
//   CUDA_CHECK(cudaHostUnregister(host));
// }

// // warpsize is a hardware property, but we query it once and cache it here, and we initialize max_threads_per_block too frmo device specs
// static int warp_size = 0;
// static int max_threads_per_block = 0;

// // Set the CUDA device and query its properties 
// extern "C" void set_cuda_device(int device) {
//   CUDA_CHECK(cudaSetDevice(device));
//   CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
//   CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
// }

// // 
// __device__ 
// inline float warp_reduce_sum(float val) {
//   for (int offset = warpSize / 2; offset > 0; offset /= 2)
//     val += __shfl_down_sync(FULL_MASK, val, offset);

//   return val;
// }

// __device__ 
// inline float warp_all_reduce_max(float val) {
// 	// Max reduction across a warp.
// 	// All threads will contain the max of all threads in the warp.
//   for (int mask = warpSize/2; mask > 0; mask /= 2) {
//     val = max(val, __shfl_xor_sync(FULL_MASK, val, mask));
//   }
//   return val;
// }

// __device__ 
// inline float block_all_reduce_max(float val) {
// 	// Max reduction across a 1-D block implemented as double warp max reduction.
// 	// All threads will contain the max of all threads in the block.
	
// 	// Will hold results of all warps.
// 	// Capacity 32 since there can be at most 32 warps in a thread.
//   __shared__ float shared[32];
//   const int wid  = threadIdx.x / warpSize;
//   const int lane = threadIdx.x % warpSize;

//   val = warp_all_reduce_max(val);

//   if (blockDim.x < warpSize) return val;
//   if (lane == 0) shared[wid] = val;

//   __syncthreads();

//   if ( wid == 0 ) {
// 	  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
//   }
//   val = warp_all_reduce_max(val);
//   if (lane == 0) shared[wid] = val;
  
//   __syncthreads();
  
//   return shared[0];
// }

// __device__ 
// inline float warp_all_reduce_sum(float val) {
// 	// Sum reduction across a warp.
// 	// All threads will contain the sum of all threads in the warp.
//   for (int mask = warpSize/2; mask > 0; mask /= 2) {
//     val += __shfl_xor_sync(FULL_MASK, val, mask);
//   }
//   return val;
// }

// __device__ 
// inline float block_all_reduce_sum(float val) {
// 	// Sum reduction across a 1-D block implemented as double warp sum reduction.
// 	// All threads will contain the sum of all threads in the block.
	
// 	// Will hold results of all warps.
// 	// Capacity 32 since there can be at most 32 warps in a thread.
//   __shared__ float shared[32];
//   const int wid  = threadIdx.x / warpSize;
//   const int lane = threadIdx.x % warpSize;

//   val = warp_all_reduce_sum(val);

//   if (blockDim.x < warpSize) return val;
//   if (lane == 0) shared[wid] = val;

//   __syncthreads();

//   if ( wid == 0 ) {
// 	  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
//   }
//   val = warp_all_reduce_sum(val);
//   if (lane == 0) shared[wid] = val;
  
//   __syncthreads();
  
//   return shared[0];
// }

// __device__
// inline float matmul_row(const float* row, const float* x, int offset, int dim) {
// 	float sum = 0.0;
// 	for (int j = offset; j < dim; j += warpSize) {
// 		float v = row[j] * x[j];
// 		sum += v;
// 	}
// 	return warp_reduce_sum(sum);
// }

// __global__
// void matmul(const float* A, const float* x, int n, int d, float* out) {
// 	// A (d,n) @ x (n,) -> out (d,)
// 	// PRECOND: Block is 1-D.
// 	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
// 	if (i >= d) return;
// 	// Since block is 1-dimensional, thread ID is same as threadIdx.x,
// 	// and warp partitions thread IDs
// 	int offset = threadIdx.x % warpSize;
// 	float rowSum = matmul_row(&A[n * i], x, offset, n);
// 	if (offset == 0) {
// 		out[i] = rowSum;
// 	}
// }

// // Attention kernels start here and this defines the attention mechanism
// __global__
// void attn(
// 	const float* kb, 
// 	const float* q, 
// 	int head_dim, 
// 	int kv_len, 
// 	int max_seq_len, 
// 	int n_heads, 
// 	float* out
// ) {
// 	int group = blockIdx.y;
// 	int t = blockIdx.x * blockDim.x + threadIdx.x;
// 	int h = blockIdx.y * blockDim.y + threadIdx.y;
// 	if (t >= kv_len || h >= n_heads) return;
	
// 	const float* query = q + h * head_dim;
// 	const float* key = kb + max_seq_len * group + head_dim * t;
// 	float score = 0.0;
// 	for (int i = 0; i < head_dim; i++) {
// 		score += query[i] * key[i];
// 	}
// }


// __global__
// void attn_softmax(
// 	const float* att, 
// 	int seq_len, 
// 	int max_seq_len, 
// 	int n_heads, 
// 	float* out
// ) {
// 	int offset = threadIdx.x;
// 	int h = blockIdx.x;
// 	int block_size = blockDim.x;
// 	if (h >= n_heads) return;
	
// 	const float* atth = att + max_seq_len * h;
// 	float* outh = out + max_seq_len * h;
	
// 	float score_max = -FLT_MAX;
// 	for (int t = offset; t < seq_len; t += block_size) {
// 		if (atth[t] > score_max) {
// 			score_max = atth[t];
// 		}
// 	}
// 	score_max = block_all_reduce_max(score_max);
// 	float score_sum = 0.0f;
// 	for (int t = offset; t < seq_len; t += block_size) {
// 		outh[t] = expf(atth[t] - score_max);
// 		score_sum += outh[t];
// 	}
// 	score_sum = block_all_reduce_sum(score_sum);
// 	for (int t = offset; t < seq_len; t += block_size) {
// 		outh[t] /= score_sum;
// 	}
// }

// __global__
// void att_mix(
// 	const float* vb, 
// 	const float* att, 
// 	int head_dim, 
// 	int n_heads, 
// 	int n_kv_heads,
// 	int seq_len, 
// 	int max_seq_len, 
// 	float* out
// ) {
// 	// PRECOND: blocks are 1-D and blockDim.x == warpSize
// 	int h = blockIdx.x;
// 	int group_size = n_heads / n_kv_heads;
// 	int g = h / group_size;
// 	int i = blockIdx.y;
// 	int offset = threadIdx.x;
	
// 	const float* atth = att + max_seq_len * h;
// 	const float* vh = vb + max_seq_len * head_dim * g;
// 	float* outh = out + head_dim * h;
	
// 	float sum = 0.0;
// 	for (int t = offset; t < seq_len; t += warpSize) {
// 		sum += vh[head_dim * t + offset] * atth[t];
// 	}
// 	sum = warp_reduce_sum(sum);
// 	if (offset == 0) outh[i] = sum;
// }

// __global__
// void rmsnorm(const float* x, const float* weight, int size, float eps, float* out) {
// 	// PRECOND: only one 1-D block is launched
// 	float rms = 0.0;
// 	int offset = threadIdx.x;
// 	for (int i = offset; i < size; i += blockDim.x) {
// 		rms += x[i] * x[i];
// 	}
// 	rms = block_all_reduce_sum(rms);
// 	rms = sqrtf(rms / size + eps);
// 	float scale = 1.0 / rms;
// 	for (int i = offset; i < size; i += blockDim.x) {
// 		out[i] = x[i] * scale * weight[i];
// 	}
// }

// __global__
// void rope(
// 	const float* x, int d, int head_dim, int pos, float theta, int rotary_dim, float* out
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
// 	if (i >= d) return;
	
// 	int j_head = i % head_dim;
// 	float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
// 	float val = pos * freq;
// 	float fcr = cosf(val);
// 	float fci = sinf(val);
	
// 	float v0 = x[i];
// 	float v1 = x[i + 1];
// 	out[i] = v0 * fcr - v1 * fci;
// 	out[i + 1] = v0 * fci + v1 * fcr;
// }

// __global__
// void add_residuals(
// 	const float* x, const float* y, int d, float* out
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (i >= d) return;
	
// 	out[i] = x[i] + y[i];
// }


// __global__
// void clip(
// 	const float* x, float v, int d, float* out
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (i >= d) return;
	
// 	out[i] = x[i] < -v ? -v : (x[i] > v ? v : x[i]);
// }

// __global__
// void glu_silu(
// 	const float* x, const float* weight, int d, float* out
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (i >= d) return;
	
// 	out[i] = weight[i] * x[i] / (1.0f + expf(-x[i]));
// }

// __global__
// void glu_gelu(
// 	const float* x, const float* weight, int d, float* out
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (i >= d) return;
	
// 	float v = x[i];
// 	out[i] =
// 		weight[i] * 
// 		0.5f * v * (1.0f + tanhf(0.797885f * (v + 0.044715f * v * v * v)));
// }

// // TODO: consolidate copy_embedding and copy_kv_entry into 1 memcpy kernel
// __global__
// void copy_embedding(
// 	const float* token_embedding_table, int dim, int token, float* out
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (i >= dim) return;
	
// 	const float* v = token_embedding_table + dim * token;
// 	out[i] = v[i];
// }

// __global__
// void copy_kv_entry(
// 	const float* in, int kv_pos, int kv_dim, float* kb
// ) {
// 	// PRECOND: grid and blocks are 1-D
// 	int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	if (i >= kv_dim) return;
	
// 	kb[kv_pos * kv_dim + i] = in[i];
// }
// template <typename T>
// void Block::_block_cuda(
// 	InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
// ) const {
// 	// TODO: support f16
// 	const Config& c = *_config;
	
// 	// attention pre-norm
// 	switch (c.norm_type) {
//     case LayerNormType::RMSNorm: {
// 	    rmsnorm<<<1, max_threads_per_block>>>(
// 				s.x(), rms_att_weight(), c.dim, c.norm_eps, s.xb()
// 			);
//       break;
//     }
//   }
  
//   int q_dim = c.n_heads * c.head_dim;
//   int kv_dim = c.n_kv_heads * c.head_dim;

//   // qkv matmuls for this position
//   matmul<<<
//     (q_dim + warp_size - 1)/warp_size, warp_size
//   >>>(wq<float>(), s.xb(), c.dim, q_dim, s.q());
//   matmul<<<
//     (kv_dim + warp_size - 1)/warp_size, warp_size
//   >>>(wk<float>(), s.xb(), c.dim, kv_dim, s.k());
//   matmul<<<
//     (kv_dim + warp_size - 1)/warp_size, warp_size
//   >>>(wv<float>(), s.xb(), c.dim, kv_dim, s.v());
  
//   // some models require clipping qkv values
//   clip<<<
// 	  (q_dim + max_threads_per_block - 1)/max_threads_per_block, 
// 	  max_threads_per_block
//   >>>(s.q(), c.qkv_clip, q_dim, s.q());
//   clip<<<
// 	  (kv_dim + max_threads_per_block - 1)/max_threads_per_block, 
// 	  max_threads_per_block
//   >>>(s.k(), c.qkv_clip, kv_dim, s.k());
//   clip<<<
// 	  (kv_dim + max_threads_per_block - 1)/max_threads_per_block,
// 	  max_threads_per_block
//   >>>(s.v(), c.qkv_clip, kv_dim, s.v());
  
//   // RoPE relative positional encoding: complex-valued rotate q and k in each head
//   rope<<<
// 	  (q_dim + max_threads_per_block - 1)/max_threads_per_block,
// 	  max_threads_per_block
//   >>>(
// 		s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim, s.q()
// 	);
// 	rope<<<
// 	  (kv_dim + max_threads_per_block - 1)/max_threads_per_block,
// 	  max_threads_per_block
//   >>>(
// 		s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim, s.k()
// 	);
	
// 	// key and value point to the kv cache
// 	float* kb = key_cache();
// 	float* vb = value_cache();
// 	copy_kv_entry<<<
// 		(kv_dim + max_threads_per_block - 1)/max_threads_per_block, 
// 		max_threads_per_block
// 	>>>(
// 		s.k(), kv_pos, kv_dim, kb
// 	);
// 	copy_kv_entry<<<
// 		(kv_dim + max_threads_per_block - 1)/max_threads_per_block, 
// 		max_threads_per_block
// 	>>>(
// 		s.v(), kv_pos, kv_dim, vb
// 	);
	
// 	// multihead attention: dot products and softmax
// 	{
// 		dim3 tpb;
// 		tpb.x = warp_size;
// 		tpb.y = max_threads_per_block / warp_size;
// 		dim3 blocks;
// 		blocks.x = (kv_len + tpb.x - 1) / tpb.x;
// 		blocks.y = (c.n_heads + tpb.y - 1) / tpb.y;
// 		attn<<<blocks, tpb>>>(
// 			kb, s.q(), c.head_dim, kv_len, c.max_seq_len, c.n_heads, s.att()
// 		);
// 		attn_softmax<<<c.n_heads, warp_size>>>(
// 			s.att(), kv_len, c.max_seq_len, c.n_heads, s.att()
// 		);
// 	}
//   // multihead attention: mix values with attention scores
// 	{
// 		dim3 tpb;
// 		tpb.x = warp_size;
// 		dim3 blocks;
// 		blocks.x = c.n_heads;
// 		blocks.y = c.head_dim;
// 		att_mix<<<blocks, tpb>>>(
// 			vb, s.att(),
// 			c.head_dim, c.n_heads, c.n_kv_heads, 
// 			kv_len, c.max_seq_len, s.xb2()
// 		);
// 	}
// 	// final matmul projection via wo, using `hb` as temp storage
// 	matmul<<<
//     (c.dim + warp_size - 1)/warp_size, warp_size
//   >>>(wo<float>(), s.att(), q_dim, c.dim, s.hb());
	
// 	// attn residual back into x
// 	add_residuals<<<
// 		(c.dim + max_threads_per_block - 1)/max_threads_per_block, 
// 		max_threads_per_block
// 	>>>(
// 		s.x(), s.hb(), c.dim, s.x()
// 	);
	
// 	// ffn pre-norm
// 	switch (c.norm_type) {
//     case LayerNormType::RMSNorm: {
// 	    rmsnorm<<<1, max_threads_per_block>>>(
// 				s.x(), rms_ffn_weight(), c.dim, c.norm_eps, s.xb()
// 			);
//       break;
//     }
//   }
	
// 	// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
//   // Note this is a feedforward with a GLU, not a simple MLP.
//   matmul<<<
//     (c.hidden_dim + warp_size - 1)/warp_size, warp_size
//   >>>(w1<float>(), s.xb(), c.dim, c.hidden_dim, s.hb());
//   matmul<<<
//     (c.hidden_dim + warp_size - 1)/warp_size, warp_size
//   >>>(w3<float>(), s.xb(), c.dim, c.hidden_dim, s.hb2());
//   switch (c.act) {
// 	  case ActivationType::GELU: {
// 		  glu_gelu<<<
// 			  (c.hidden_dim + max_threads_per_block - 1)/max_threads_per_block, 
// 			  max_threads_per_block
// 		  >>>(
// 				s.hb(), s.hb2(), c.hidden_dim, s.hb()
// 			);
// 		  break;
// 	  }
// 	  case ActivationType::SILU: {
// 		  glu_silu<<<
// 			  (c.hidden_dim + max_threads_per_block - 1)/max_threads_per_block, 
// 			  max_threads_per_block
// 		  >>>(
// 				s.hb(), s.hb2(), c.hidden_dim, s.hb()
// 			);
// 		  break;
// 	  }
//   }
  
//   matmul<<<
//     (c.dim + warp_size - 1)/warp_size, warp_size
//   >>>(w2<float>(), s.hb(), c.hidden_dim, c.dim, s.xb2());
  
// 	// ffn residual back into x
// 	add_residuals<<<
// 		(c.dim + max_threads_per_block - 1)/max_threads_per_block,
// 		max_threads_per_block
// 	>>>(
// 		s.x(), s.xb2(), c.dim, s.x()
// 	);
// }

// void Model::_forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode) {
// 	const Config& c = *config;
// 	if (c.weight_dtype != DType::F32) {
// 		assert(false && "unsupported weight dtype for CUDA");
// 	}
	
// 	copy_embedding<<<
//     (c.dim + max_threads_per_block - 1)/max_threads_per_block,
// 		max_threads_per_block
// 	>>>(
// 		static_cast<float*>(token_embedding_table), c.dim, token, s.x()
// 	);
	
// 	// TODO: attention sinks
// 	int kv_pos = pos % c.max_seq_len;
// 	int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
	
// 	// forward all layers in order
// 	// TODO understand what the kv_sink argument does
// 	for (auto b : blocks) {
// 		b->block(s, pos, 0, kv_pos, kv_len);
// 	}

//   if (mode == InferenceMode::HYDRATE_KV_CACHE) {
//     // only hydrate the KV cache and don't compute output logits
// 	  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
//     return;
//   }
	
// 	// final layer norm
// 	switch (c.norm_type) {
// 		case LayerNormType::RMSNorm: {
// 			rmsnorm<<<1, max_threads_per_block>>>(
// 				s.x(), rms_final_weight, c.dim, c.norm_eps, s.x()
// 			);
// 			break;
// 		}
// 	}
	
// 	// classifier into logits
// 	switch (c.weight_dtype) {
//     case DType::F32: {
// 	    matmul<<<
// 		    (c.vocab_size + warp_size - 1)/warp_size, warp_size
// 	    >>>(static_cast<float*>(wcls), s.x(), c.dim, c.vocab_size, s.logits());
//       break;
//     }
//     default: {
//       assert(false && "unsupported weight dtype for CUDA");
//     }
//   }
	
// 	CUDA_CHECK(cudaDeviceSynchronize()); // After this, s.logits contains logits of output token
// 	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
// }