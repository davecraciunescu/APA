#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>


cudaError_t multMatCuda(int* result, int* a, int* b, int size);

__global__ void multMat(int* result, int* a, int* b, int size)
{
	// Identifier of the corresponding executino at the GPU. It will be
    // obtained from the block's ID and the corresponging thread's ID
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < size && row < size)
    {
        for(int i = 0; i < size; i++)
        {
            result[row * size + col] += a[col * size + i] * b[i * size + col];
        }
    }
}

int main()
{
    const int size = 16;

	int* a      = (int*)calloc(size * size, sizeof(int));  
	int* b      = (int*)calloc(size * size, sizeof(int));  
	int* result = (int*)calloc(size * size, sizeof(int));  
    
    // Data loaded into the arrays
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            a[i * size + j] = 1;
            b[i * size + j] = 2;
        }
    }

    printf("Data Initialized\n");

    // Kernel method execution
    cudaError_t err = multMatCuda(result, a, b, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "multMat failed!\n");
		return 1;
	}
    
    // Operation is printed
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            printf("| ");
            
            for(int k = 0; k < size; k++)
            {
                switch(j)
                {
                    case 0:
                        printf("%d", a[i * size + k]);
                        break;
                    
                    case 1:
                        printf("%d", b[i * size + k]);
                        break;
                
                    case 2:
                        printf("%d", result[i * size + k]);
                        break;
                }
                if(k != size - 1) printf(", ");
            }

            if(i != (int) (size / 2) && j != 2) {
                printf(" |   ");
            } else if(j == 2) {
                printf(" |");
            } else {
                switch(j)
                {
                    case 0:
                        printf(" | * ");
                        break;
                    
                    case 1:
                        printf(" | = ");
                        break;
                    case 2:
                        printf(" |");
                }
            }

        }

        printf("\n");
    }

	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}

cudaError_t multMatCuda(int* result, int* a, int* b, int size)
{
    int TILE_WIDTH = 1;
	// As many cuda errors are going to be treated, it is better to initialize
	// here the error variable
	cudaError_t err;
	// Variables used to operate in GPU mem
	int* dev_result = 0;
	int* dev_a      = 0;
	int* dev_b      = 0;

    dim3 dimGrid(size / TILE_WIDTH, size / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	
    // Select GPU where the threads are to be executed
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		goto Error;
	}

	// Allocate memory for GPU's variables
	err = cudaMalloc((void**)&dev_result, size * size * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_a, size * size * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	err = cudaMalloc((void**)&dev_b, size * size * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMaloc failed!");
		goto Error;
	}

	// Transfers data from host variables to GPU variables
	err = cudaMemcpy(dev_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	err = cudaMemcpy(dev_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


    // Launch kernel to multiply 2 16x16 matrices
	multMat<<<dimGrid, dimBlock>>>(dev_result, dev_a, dev_b, size);

	// Checks whether the execution in the GPU has been completed correctly
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "multMat launch failed: %s\n",
			cudaGetErrorString(err));
		goto Error;
	}

	// Waits for the kernel to finish and checks whether theres has been any
	// errors or not
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multMat!\n", err);
		goto Error;
	}

	err = cudaMemcpy(result, dev_result, size * size * sizeof(int), cudaMemcpyDeviceToHost);
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

