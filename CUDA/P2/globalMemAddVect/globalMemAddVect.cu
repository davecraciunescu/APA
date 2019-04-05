#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

cudaError_t addVectCuda(int* result, int* a, int* b, int size);

__global__ void addVect(int* result, int* a, int* b, int size)
{
	// Identifier of the corresponding executino at the GPU. It will be
    // obtained from the block's ID and the corresponging thread's ID
    int pos = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if(pos < size) result[pos] = a[pos] + b[pos];
}

int main()
{
    const int size = 24;

	int a[size];  
	int b[size]; 
	int result[size];
    
    // Data loaded into the arrays
    for(int i = 0; i < size; i++)
    {
        a[i] = -i;
        b[i] = i + i;
    }
	
    // Kernel method execution
    cudaError_t err = addVectCuda(result, a, b, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "addVectCuda failed!\n");
		return 1;
	}
    
    // Operation is printed
    // Print A
    printf("  {");
    for(int i = 0; i < size; i++)
    {
        printf("%d", a[i]);
        if(i != size - 1) printf(", ");
    }
    printf("}\n+ {");
    // Print B
    for(int i = 0; i < size; i++)
    {
        printf("%d", b[i]);
        if(i != size - 1) printf(", ");
    }
    printf("}\n");
    printf("---------------------------------------------------------------------------------------------");
    printf("\n  {");
    // Print Result
    for(int i = 0; i < size; i++)
    {
        printf("%d", result[i]);
        if(i != size - 1) printf(", ");
    }
    printf("}");

	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}

cudaError_t addVectCuda(int* result, int* a, int* b, int size)
{
	// As many cuda errors are going to be treated, it is better to initialize
	// here the error variable
	cudaError_t err;
	// Variables used to operate in GPU mem
	int* dev_result = 0;
	int* dev_a = 0;
	int* dev_b = 0;

	// Select GPU where the threads are to be executed
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		goto Error;
	}

	// Allocate memory for GPU's variables
	err = cudaMalloc((void**)&dev_result, size * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	// Transfers data from host variables to GPU variables
	err = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	err = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Launch kernel to add three values
	addVect <<<3, 8>>>(dev_result, dev_a, dev_b, size);

	// Checks whether the execution in the GPU has been completed correctly
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "addVect launch failed: %s\n",
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

	err = cudaMemcpy(result, dev_result, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_result);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return err;
}

