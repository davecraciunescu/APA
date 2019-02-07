#include <iostream>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

/**
    Three integer addition (CUDA Kernel) on global memory.
*/
__global__ void Addition_kernel(int* a, int* b, int* c, int* d)
{
    // Perform the Addition operation.
    *d = *a + *b + *c;
}

/**
    Host main routine.
*/
int main()
{
    // Error code to check return values for CUDA calls.
    cudaError_t error = cudaSuccess;

    // ------------------- HOST MEMORY VARIABLE ALLOCATION --------------------
    int* h_A = (int*) malloc(sizeof(int));
    int* h_B = (int*) malloc(sizeof(int));
    int* h_C = (int*) malloc(sizeof(int));
    int* h_D = (int*) malloc(sizeof(int));

    // Verify the allocations happened.
    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // ------------------ DEVICE MEMORY VARIABLE ALLOCATION -------------------
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;
    int* d_D = nullptr;

    error = cudaMalloc((void**) &d_A, sizeof(int));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable A, (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**) &d_B, sizeof(int));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable B, (error code %s)\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**) &d_C, sizeof(int));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable C, (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**) &d_D, sizeof(int));
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for Variable D, (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
   }

    // ------------------------------------------------------------------------

    // Give the memory random values.
    *h_A = 3;
    *h_B = 5;
    *h_C = 1;

    // ------------------- COPY HOST MEMORY TO DEVICE MEMORY ------------------

    error = cudaMemcpy(d_A, h_A, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable A from host to device (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable B from host to device (error code %s)\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_C, h_C, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable C from host to device (error code %s)\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ---------------------- LAUNCH THE ADDITION KERNEL ----------------------
    Addition <<<1,1>>> (d_A, d_B, d_C, d_D);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch the Addition Kernel (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);

    }

    // --------------- COPY DEVICE MEMORY TO HOST MEMORY ----------------------
    error = cudaMemcpy(&h_D, d_D, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // ------------------------ PRINT RESULT ON SCREEN ------------------------
    printf("Result of operation is: %d\n", &h_D); 

    // -------------------------- FREE DEVICE MEMORY --------------------------
    error = cudaFree(d_A);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Variable A (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_B);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Variable B (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_C);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Variable C (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaFree(d_D);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device Variable D (error code %s)!\n",
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
 
    // --------------------------- FREE HOST MEMORY ---------------------------
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return EXIT_SUCCESS;
}

#endif // ADDITION.H
