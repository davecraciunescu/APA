#include <stdio.h>
#include <cstddef>
#include <iostream>

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
* wA is A's width and wB is B's width.
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float* C, float* A, float* B, int wA, int wB)
{
    // Indexes of the first sub-matrices processed by the block.
    int aBegin  = wA * BLOCK_SIZE * blockIdx.y;
    int bBegin  = BLOCK_SIZE * blockIdx.x;

    // Step sizes used to iterate through the different matrices.
    int aStep   = BLOCK_SIZE;
    int bStep   = BLOCK_SIZE * wB;

    // Indexes of the last sub-matrices procesed by the block.
    int aEnd    = aBegin + wA - 1;

    // Stores the element of the block sub-matrix computed by the thread.
    float Csub  = 0;

    // ------------------------------------------------------------------------

    // Compute the block sub-matrix.
    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {
        // Shared memory arrays used to store the sub-matrices.
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory.
        As[threadIdx.y][threadIdx.x] = A[a + (wA * threadIdx.y) + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b + (wB * threadIdx.y) + threadIdx.x];

        // Sync to make sure the matrices are loaded.
        __syncthreads();

        // Multiply the two matrices together.
        #pragma unroll // -> Optimize loop execution.
        for (int k = 0; k < BLOCK_SIZE, k++)
        {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Sync before loading new data.
        __syncthreads();
    }

    // ------------------------------------------------------------------------

    // Write to device memory.

    int c = (wB * BLOCK_SIZE * blockIdx.y) + (BLOCK_SIZE * blockIdx.x);
    C[c + (wB * threadIdx.y) + threadIdx.x] = Csub;
}

/**
* Data Initializer.
*/
void constantInit (float* data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
* Run a simple test of CUDA matrix multiplication.
*/
int matrixMultiply (int argc, char** argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices.
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C = (float*) malloc(mem_size_C);

    // Initiate host memory.
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, 0.01f);

    // Allocate device memory.
    float *d_A, *d_B, *d_C;

    // ------------------------------------------------------------------------

    if (h_C == nullptr)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    cudaError_t error;

    // Allocate CUDA Memory for A.
    error = cudaMalloc((void**) &d_A, mem_size_A);
    if (error != cudaSuccess)
    {
        cudaMallocError("d_A", error);
    }

    // Allocate CUDA Memory for B.
    error = cudaMalloc((void**) &d_B, mem_size_B);
    if (error != cudaSuccess)
    {
        cudaMallocError("d_B", error);
    }

    // Allocate CUDA Memory for C.
    error = cudaMalloc((void**) &d_C, mem_size_C);
    if (error != cudaSuccess)
    {
        cudaMallocError("d_C", error);
    }

    // ------------------------------------------------------------------------
    
    // Copy host memory to device.
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaMemcpyError("d_A", "h_A", error);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaMemcpyError("d_B", "h_B", error);
    }

    // ------------------------------------------------------------------------

    // Setup execution parameters.
    dim3 threads(block_size, block_size);
    dim3 grid( (dimsB.x / threads.x), (dimsA.y / threads.y) );

    // ------------------------------------------------------------------------

    // Execute kernel.
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
        matrixMulCUDA<16> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    // ------------------------------------------------------------------------

    // Copy result from device to host.
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        cudaMemcpyError("h_C", "d_C", error);
    }

    // ------------------------------------------------------------------------

    // Clean up host memory.
    free(h_A);
    free(h_B);
    free(h_C);

    // Clean up device memory.
    error = cudaFree(d_A);
    if (error != cudaSuccess)
    {
        cudaFreeError("d_A", error);
    }

    error = cudaFree(d_B);
    if (error != cudaSuccess)
    {
        cudaFreeError("d_B", error);
    }

    error = cudaFree(d_C);
    if (error != cudaSuccess)
    {
        cudaFreeError("d_C", error);
    }
}

/**
* Program Main.
*/
int main (int argc, char **argv)
{
    // PRINT STARTING MESSAGE:

    int block_size = 32;

    dim3 dimsA(16, 16, 1);
    dim3 dimsB(16, 16, 1);

    // Set up arbitrary parameters.
    dimsA.x = 16;
    dimsA.y = 16;

    dimsB.x = 16;
    dimsB.y = 16;

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}

// Error Handling Methods.
void cudaFreeError(std::string id, cudaError_t msg)
{
    std::cout <<"Failed to free device from "<< id <<" (error code "<<
        cudaGetErrorString(msg) << ")!\n";
    exit(EXIT_FAILURE);
}

void cudaMemcpyError(std::string from, std::string to, cudaError_t msg)
{
    std::cout <<"cudaMemcpy ("<< from << to <<") returned error "<<
    cudaGetErrorString(msg) <<"(code "<< error <<"), line ("<<__LINE__ <<")\n";
    exit(EXIT_FAILURE);
}

void cudaMallocError(std::string id, cudaError_t msg)
{
    std::cout <<"cudaMalloc "<< id <<"returned error "<< cudaGetErrorString(msg)
    <<" (code "<< error <<"), line ("<<__LINE__<<")\n";
    exit(EXIT_FAILURE);
}
