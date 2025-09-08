#include <iostream>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// As the matrix size decreases, we get our CPU time to be faster than GPU.
// Why is that? 
// Latency is the time taken for data to travel from one point to the next 
// Bandwidth is the maximum amount of data tha can be sent nad recieved in a given time 
// 


// 
// CPU sq_mat_mul took 0.378262 sec
// C[0,0] = 1024
//GPU sq_mat_mul took 0.0325674 sec


#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)
#define MAX_NUM 10 
#define MIN_NUM -10 



// remember that C always holds memory in a contiguous manner
// So here float* A is N1,N2,N3,N4,N5, etc. in a straight line 
// And if we are only talking about square matrices, then this N
// bounds the rows, so after N numbers a new row starts.
// The formula for index is r * N + c, where c is the column number
// so let's say that we want to see position [1,1] in a 2 x 2 matrix 
// The matrix will be stored as N1,N2,N3,N4, with N=2. 
// Hence we go in the second row 1 * 2 + 1 = 3, A[3] = N4
// it is just a little weird with 0 based indexing
void sq_mat_mul_cpu(float* A, float* B, float* C, int N) {
    // loop over rows 
    // All we are doing is that we are iterating over C
    // for row in C
    for (int r = 0; r < N; r++) {
        // loop over the columns in C
        for (int c = 0; c < N; c++) {
            // this is the value at c[i,j]
            float value = 0;

            for (int k = 0; k < N; k++) {
                value += A[r*N+k] * B[k*N+c];
            }

            // in the rth row and cth column we put value 
            // row*N + column
            C[r*N+c] = value;
        }
    }
}

__global__  void sq_mat_mul_kernel(float* A, float* B, float* C, int N) {
    // Now let's understand the indexing of the language 
    // gridDim.x gives the number of blocks in teh x dimension 
    // gridDim.y gives the number of blocks in the y dimension 
    // blockDim.x gives the number of threads in the block in X direction 
    // blockDim.y gives the number of threads in the block in Y direction
    // blockIdx.x gives the x-coordinate of the block that this thread belongs to 
    // blockIdx.y gives the y-coordinate of the block that this thread belongs to 
    // threadIdx.x gives the x-coordinate of the thread inside of it's respective block 
    // threadIdx.y gives the y-coordinate of the thread inside of it's respective block 
    
    // let's first assign the row number of the thread
    // In sum, the we multiply the number of threads per block by which block we are in and then add the thread we are on  
    int r = blockDim.y*blockIdx.y + threadIdx.y;
    int c = blockDim.x*blockIdx.x + threadIdx.x;

    if (r < N && c < N) {
        // Here we intiliaze our dot product sum 
        float dot_product_sum = 0;

        for (int k = 0; k < N; k++) {
            // We want the rth row of A to be dot prod with cth column of B
            // we want row*N + column  
            dot_product_sum += A[r*N + k]*B[k*N + c];
        }
        C[r*N + c] = dot_product_sum;
    }



}

#define TILE_WIDTH 32

__global__
void sq_mat_mul_tiled_kernel(float* A, float*B, float* C, int N) {
    // here we assume that we have square blocks 
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Ensure N%TILE_WIDTH == 0
    // We make sure that the width is divisible 
    assert(N % TILE_WIDTH == 0);

    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int r = blockDim.y*blockIdx.y + dy;
    int c = blockDim.x*blockIdx.x + dx;
    
    // Here we allocate the shared memory 
    // Keep in mind this is just allocated by the block and not by the thread 
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];
    
    float value = 0;

    for (int phase = 0; phase < N/TILE_WIDTH; phase++) {

        // This is the most complicated part of the code 
        // So we only write to the thread location in the shared tiles 
        // So thread 0,0 only writes to sh_A[0][0] and sh_B[0][0]
        sh_A[dy][dx] = A[r*N + phase*TILE_WIDTH+dx];
        sh_B[dy][dx] = A[(phase*TILE_WIDTH+dy)*N + c];

        __syncthreads();

        // Here we are performing the partial dot products 
        // BUT WE ARE ACCESSING MEMORY STORED BY OTHER THREADS!!!!!
        // This is the teamwork aspect brought up by blocks!
        // This is why we need synch threads before we do anything
        for (int k =0; k < TILE_WIDTH; k++) {
            value += sh_A[dy][k]*sh_B[k][dx];
        }
        // Before we move to the next loop we need to make sure that all tiles 
        __syncthreads();

        
    }

    C[r*N+c] = value;
}

__global__
void mat_mul_tiled_kernel(float* A, float*B, float* C, int N) {
    // here we assume that we have square blocks 
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    
    // Ensure N%TILE_WIDTH == 0
    // We make sure that the width is divisible 
    assert(N % TILE_WIDTH == 0);

    int dx = threadIdx.x;
    int dy = threadIdx.y;
    int r = blockDim.y*blockIdx.y + dy;
    int c = blockDim.x*blockIdx.x + dx;
    
    // Here we allocate the shared memory 
    // Keep in mind this is just allocated by the block and not by the thread 
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];
    
    float value = 0;

    for (int phase = 0; phase < N/TILE_WIDTH; phase++) {

        // This is the most complicated part of the code 
        // So we only write to the thread location in the shared tiles 
        // So thread 0,0 only writes to sh_A[0][0] and sh_B[0][0]
        sh_A[dy][dx] = A[r*N + phase*TILE_WIDTH+dx];
        sh_B[dy][dx] = A[(phase*TILE_WIDTH+dy)*N + c];

        __syncthreads();

        // Here we are performing the partial dot products 
        // BUT WE ARE ACCESSING MEMORY STORED BY OTHER THREADS!!!!!
        // This is the teamwork aspect brought up by blocks!
        // This is why we need synch threads before we do anything
        for (int k =0; k < TILE_WIDTH; k++) {
            value += sh_A[dy][k]*sh_B[k][dx];
        }
        // Before we move to the next loop we need to make sure that all tiles 
        __syncthreads();

        
    }

    C[r*N+c] = value;
}

int main() {
    int Z = 256;
    std::vector<float> D(Z*Z), E(Z*Z), F(Z*Z);

    for (int i = 0; i < Z*Z; i++) {
        D[i] = 1.0f;
        E[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    sq_mat_mul_cpu(D.data(), E.data(), F.data(), Z);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "CPU sq_mat_mul took " << elapsed.count() << " sec\n";

    // Optional: check one result
    std::cout << "C[0,0] = " << F[0] << "\n";

    // Generate NxN square matrices A and B
    int N = 256;
    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));
    float* C = (float*)malloc(N*N*sizeof(float));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            B[i*N+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }

    // First we define the device variables that will be passed into the GPU
    float* d_A; 
    float* d_B; 
    float* d_C;

    // Next we assign the memory needed for each of these vars 
    // keep in mind that the output of cudaMalloc is a cudaError_t type 
    cudaError_t err_A = cudaMalloc((void**) &d_A, N*N*sizeof(float));
    CUDA_CHECK(err_A);
    cudaError_t err_B = cudaMalloc((void**) &d_B, N*N*sizeof(float));
    CUDA_CHECK(err_B);
    cudaError_t err_C = cudaMalloc((void**) &d_C, N*N*sizeof(float));
    CUDA_CHECK(err_C);

    // Now we copy the arrays onto the GPU memory with cudmemcpy 
    cudaError_t err_A_ = cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_A_);
    cudaError_t err_B_ = cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err_B_);
   
    // execute the kernel 
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(N/32.0), ceil(N/32.0), 1);
    auto start_ = std::chrono::high_resolution_clock::now();
    sq_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);
    auto end_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ = end_ - start_;
    std::cout << "GPU sq_mat_mul took " << elapsed_.count() << " sec\n";

    // copy back the results
    cudaError_t err_C_ = cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err_C_);

    auto start_tiled = std::chrono::high_resolution_clock::now();
    sq_mat_mul_tiled_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);
    auto end_tiled = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tiled = end_tiled - start_tiled;
    std::cout << "GPU sq_mat_mul with tiling took " << elapsed_tiled.count() << " sec\n";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
     
    // Here we can get the properties of a device 
    cudaDeviceProp dev_prop; 
    cudaGetDeviceProperties(&dev_prop, 0);
    std::cout << "\nThe maximum threads per block is:" << dev_prop.maxThreadsPerBlock;
    std::cout << "\nThe maximum SMs in GPU is:" << dev_prop.multiProcessorCount;
    std::cout << "\nThe maximum clockrate of each SM is:" << dev_prop.clockRate;
    std::cout << "\nThe maximum warp size is:" << dev_prop.warpSize << "\n";
}