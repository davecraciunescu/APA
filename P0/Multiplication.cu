#include <stdio.h>
#include <cstddef>
#include <assert.h>
#include <cuda_runtime.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>

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
    
    // Indexes of the last sub-matrices processed by the block.
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
        for (int k = 0; k < BLOCK_SIZE; k++)
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
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);
    
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

    // Allocate CUDA memory for A.
    error = cudaMalloc((void**) &d_A, mem_size_A);
    if(error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line (%d)\n",
            cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Allocate CUDA memory for B.
    error = cudaMalloc((void**) &d_B, mem_size_B);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line (%d)\n",
            cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Allocate CUDA memory for C.
    error = cudaMalloc((void**) &d_C, mem_size_C);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line (%d)\n",
            cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    // Copy host memory to device.
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A, h_A) returned error %s (code %d), line(%d)\n",
            cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy host memory to device.
    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B, h_B) returned error %s (code %d), line(%d)\n",
            cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    // Setup execution parameters.
    dim3 threads(block_size, block_size);
    dim3 grid( (dimsB.x / threads.x), (dimsA.y / threads.y) );

    // ------------------------------------------------------------------------

    // Create and start timer.
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation.
    if (block_size == 16)
    {
        matrixMulCUDA<16> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
        matrixMulCUDA<32> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("Done.\n");

    // ------------------------------------------------------------------------

    cudaDeviceSynchronize();

    // Allocate CUDA events used for timing.
    cudaEvent_t start;

    error = cudaEventCreate(&start);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;

    error = cudaEventCreate(&stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    error = cudaEventRecord(start, nullptr);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute kernel.
    int nIter = 300;

    for (int j = 0; j < nIter; j++)
    {
        if (block_size == 16)
        {
            matrixMulCUDA<16> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDA<32> <<<grid, threads>>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record stop event.
    error = cudaEventRecord(stop, nullptr);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    // Wait for the stop event to complete.
    error = cudaEventSynchronize(stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code%s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Calculate times.
    float msecTotal = 0.0f;

    error = cudaEventElapsedTime(&msecTotal, start, stop);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    // Copy result from device to host.
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C, d_C) returned error %s (code %d), line(%d)\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Clean up memory.
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
}

/**
* Program Main
*/
int main (int argc, char **argv)
{
    printf("CUDA Matrix Multiplication - Starting...\n");

    if (checkCmdLineFlag(argc, (const char**) argv, "help") ||
        checkCmdLineFlag(argc, (const char**) argv, "?"))
    {
        printf("Usage   -device=n (n >= 0 for deviceID)\n");
        printf("        -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("        -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("    Note: Outer matrix dimensions of matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // Pick best possible CUDA device.
    int dev = findCudaDevice(argc, (const char**) argv);

    int block_size = 32;
    
    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    // Width of Matrix A
    if (checkCmdLineFlag(argc, (const char**) argv, "wA"))
    {
        dimsA.x = getCmdLineArgumentInt(argc, (const char**) argv, "wA");
    }

    // Height of Matrix A.
    if (checkCmdLineFlag(argc, (const char**) argv, "hA"))
    {
        dimsA.y = getCmdLineArgumentInt(argc, (const char**) argv, "hA");
    }

    // Width of Matrix B.
    if (checkCmdLineFlag(argc, (const char**) argv, "wB"))
    {
        dimsB.x = getCmdLineArgumentInt(argc, (const char**) argv, "wB");
    }

    // Height of Matrix B.
    if (checkCmdLineFlag(argc, (const char**) argv, "hB"))
    {
        dimsB.y = getCmdLineArgumentInt(argc, (const char**) argv, "hB");
    }

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
                dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), Matrix(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}
