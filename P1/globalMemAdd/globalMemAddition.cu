#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

cudaError_t addThreeNumbersCuda(int* result, int* a, int* b, int* c);

__global__ void addThreeNums(int* result, int* a, int* b, int* c)
{
	*result = *a + *b + *c;
}

int main()
{
	int a = 5;
	int b = 10;
	int c = 15;
	int result = 0;

	cudaError_t err = addThreeNumbersCuda(&result, &a, &b, &c);
	if (err != cudaSuccess) {
		fprintf(stderr, "addThreeNumbersCuda failed!\n");
		return 1;
	}

	printf("%d + %d + %d = %d\n", a, b, c, result);

	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}

cudaError_t addThreeNumbersCuda(int* result, int* a, int* b, int* c)
{
	// As many cuda errors are going to be treated, it is better to initialize
	// here the error variable
	cudaError_t err;
	// Variables used to operate in GPU mem
	int* dev_result = 0;
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	// Select GPU where the threads are to be executed
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		goto Error;
	}

	// Allocate memory for GPU's variables
	err = cudaMalloc((void**)&dev_result, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_a, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_b, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_c, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	// Transfers data from host variables to GPU variables
	err = cudaMemcpy(dev_a, a, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	err = cudaMemcpy(dev_b, b, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	err = cudaMemcpy(dev_c, c, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
    
    // Launch kernel to add three values
	addThreeNums <<<1, 1>>>(dev_result, dev_a, dev_b, dev_c);

	// Checks whether the execution in the GPU has been completed correctly
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "addThreeNums launch failed: %s\n",
			cudaGetErrorString(err));
		goto Error;
	}

	// Waits for the kernel to finish and checks whether theres has been any
	// errors or not
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addThreeNums!\n", err);
		goto Error;
	}

	err = cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_result);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return err;
}

