#include <cstddef>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

__device__ float calculate_gpu(float x)
{
    return cos(pow(x,2) / 2) * pow(x, 3) * sin(pow(x,2)) / pow(x,2) + 3;
}

float calculate_cpu(float x)
{

    return cos(pow(x,2) / 2) * pow(x, 3) * sin(pow(x,2)) / pow(x,2) + 3;
}

__global__ void traps(float* a, float* b, float* h, float* result)
{
    // Set execution variables.
    const int             n = 13500;
    const int    numThreads = 256;

    // Shared partial result.
    __shared__ float part[numThreads];

    // ------------------------------------------------------------------------

    int iter = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0;

    while (iter < n)
    {
        if (iter != 0)
        {
            temp += calculate_gpu(*a + (*h * iter)); 
        }

        iter += blockDim.x * gridDim.x; // Iterate.
    }

    part[threadIdx.x] = temp;

    __syncthreads();

   // ------------------------------------------------------------------------- 

    int i = blockDim.x / 2;
    
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            part[threadIdx.x] += part[threadIdx.x + i];
        }

        __syncthreads();
        
        i /= 2;
    }

    if (threadsIdx.x == 0)
    {
        result[blockIdx.x] = part[0];
    }

    // ------------------------------------------------------------------------
}

int main (int argc, char** argv)
{
    const float*           a = 1.5f;
    const float*           b = 2.78f;    
    const int  blocksPerGrid = min(32, (N + numThreads - 1) / numThreads);
    const int     numThreads = 256;
    const int              n = 13000;

    float* h = (b - a) / n;
    
    float* cpu_result;
    float* gpu_result;

    cpu_result = (float*) malloc(blocksPerGrid * sizeof(float));

    
    cudaError_t error;

    // Allocate CUDA memory.
    error = cudaMalloc((void**) &gpu_result, blocksPerGrid * sizeof(float));
    if (error != cudaSuccess)
    {
        cudaMallocError("gpu result", error);                
    }

    traps <<<blocksPerGrid, numThreads>>> (a, b, h, gpu_result);
    
    // Copy to Host.
    error = cudaMemcpy(cpu_result, gpu_result, blocksPerGrid * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        cudaMemcpyError("cpu result", "gpu result", error); 
    }

    float partialSum = (calculate_cpu(a) + calculate_gpu(b)) / 2.0f;

    for (int i = 0; i < blocksPerGrid; i++)
    {
        partialSum += cpu_result[i];
    }

    partialSum *= h;

    std::cout << "Resultado Integral con GPU: " << partialSum << std::endl;
    partialSum = (calculate_cpu(a) + calculate_cpu(b)) / 2.0f;

    for (int i=1; i < n; i++)
    {
        partialSum += calculate_cpu(a + i * &h);
    }

    partialSum *= &h;

    std::cout << "Resultado Integral con CPU: " << partialSum << std::endl;
   
    // Free memory.
    error = cudaFree(gpu_result);
    if (error != cudaSuccess)
    {
        cudaFreeError("gpu result", error);
    }
    
    free(cpu_result);
    return 0;
}

// Auxiliary error handling methods.
// ----------------------------------------------------------------------------

void cudaMallocError(std::string id, cudaError_t msg)
{
    std::cout << "cudaMalloc " << id <<" returned error "<<
    cudaGetErrorString(msg) << " (code "<< error <<"), line ("<< __LINE__
    <<")\n";
    exit(EXIT_FAILURE);
}

void cudaMemcpyError(std::string from, std::string to, cudaError_t msg)
{
    std::cout << "cudaMemcpy ("<< from << to <<") returned error"<<
    cudaGetErrorString(msg) <<"(code "<< error <<"), line ("<< __LINE__ <<")\n";
    exit(EXIT_FAILURE);
}

void cudaFreeError(std::string id, cudaError_t msg)
{
    std::cout <<"Failed to free device from "<< id <<" (error code " <<
        cudaGetErrorString(msg) << ")!\n";
    exit(EXIT_FAILURE);
}
