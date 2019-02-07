#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

/**
  Vector addition (CUDA Kernel) on the device: C = A + B
  lA is A's length and lB is B's length.
*/
__global__ void
vectorAdd(const float* A, const float* B, float* C, int numElements)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (i < numElements)
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
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	printf("Vector Addition of %d elements\n", numElements);

	// Allocate host memory for vectors.
	float* h_A = (float*) malloc(size);
	float* h_B = (float*) malloc(size);
	float* h_C = (float*) malloc(size);

	// Verify that allocations happened.
	if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	
	// ---------------------------------------------------------------------------

	// Initialize the host input vectors.
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	// ---------------------------------------------------------------------------

	// Allocate device memory for vectors.
	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;

	error = cudaMalloc((void**) &d_A, size);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**) &d_B, size);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void**) &d_C, size);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// ---------------------------------------------------------------------------

	// Copy host memory to device memory.
	error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// ---------------------------------------------------------------------------

	// Launch the Vector ADD Cuda Kernel
	int threadsPerBlock     = 256;
	int blocksPerGrid	= (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA Kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd <<<blocksPerGrid, threadsPerBlock>>> (d_A, d_B, d_C, numElements);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}	

	// ---------------------------------------------------------------------------

	// Copy device memory to host.
	printf("Copy output data from the CUDA device to the host memory.\n");

	error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n",
			cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// ---------------------------------------------------------------------------

	// Verify that the result vector is correct.
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i] - h_C[i]) < 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test Passed.\n");

	// ---------------------------------------------------------------------------

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
