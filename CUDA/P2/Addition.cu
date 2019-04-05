#include <stdio.h>
#include <cstddef>
#include <iostream>

/**
 Vector Addition (CUDA Kernel) on the device: C = A + B
 lA is A's length and lB is B's length.
*/
__global__ void vectorAddCUDA(float* C, float* A, float* B, int numElem)
{
    int i = (blockDim.x * blockIdx.x) + threadIdx.x);

    if (i < numElem)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 Host main routine.
*/
int main (void)
{
    // Error code to check return values for CUDA calls.
    cudaError_t error = cudaSuccess;

    // Print the vector length to be used and compute its size.
    int numElem = 12345;
    size_t size = numElem * sizeof(int);
    
    // ------------------------------------------------------------------------
    // Host memory variable allocation.
    int* h_A = (float*) malloc(size_t);
    int* h_B = (float*) malloc(size_t);
    int* h_C = (float*) malloc(size_t);
    
    // Verify the allocation happened.
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------
    // Initialize the host input vectors.
    for (int i = 0; i < numElem; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // ------------------------------------------------------------------------
    // Device memory allocation.
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    error = cudaMalloc((void**) &d_A, size);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable A, (error code %s!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**) &d_B, size);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable B, (error code %s!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**) &d_C, size);
    if (error != cudaSucess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable C, (error code %s!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
   }

    // ------------------------------------------------------------------------
    // Copy host memory to device memory.
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    // Launch the vector ADD Cuda Kernel.
    int threadsPerBlock = 8;
    int blocksPerGrid   = 3;

    vectorAdd <<< blocksPerGrid, threadsPerBlock>>> (d_A, d_B, d_C, numElem);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------

    // Copy device memory to host.
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", 
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------------
    
    // Free Device memory.
    error = cudaFree(d_A);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_B);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_C);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Free host memory.
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done.\n");

    return(0);
}
