#include <stdio.h>
#include <cstddef>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_function.h>
#include <helper_cuda.h>

/**
  Matrix multiplication (CUDA Kernel) on the device: C = A * B
  wA is A's width and wB is B's width.
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float* C, float* A, float* B, int wA, int wB)
{
	// Indexed of the first sub-matrices processed by the block.
	int aBegin	= wA * BLOCK_SIZE * blockIdx.y;
	int bBegin	= BLOCK_SIZE * blockIdx.x;

	// Step sizes used to iterate through the different matrices.
	int aStep	= BLOCK_SIZE;
	int bStep	= BLOCK_SIZE * wB;

	// Indexed of the last sub-matrices processed by the block.
	int aEnd	= aBegin + wA - 1;

	// Stores the element of the block sub-matrix computed by the thread.
	float Csub 	= 0;

	// ---------------------------------------------------------------------------

	// Compute the block sub-matrix.
	for (int a = aBegin, b = bBehin;
		a <= aEND;
		a += aStep, b += bStep)
	{
		// Shared memory arrays used to store the sub-matrices.
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];


	}
}
