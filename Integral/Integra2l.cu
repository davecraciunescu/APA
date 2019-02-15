#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cutil.h>

__device__ float calculate_gpu(float x)
{
    return __sin(1/x) * pow(x,3) / (x+1) * (x+2);
}

float calculate_cpu(float x)
{
    return sin(1/x) * pow(x,3) / (x+1) * (x+2);
}

__global__ void traps(float* a, float* b, float* h, float* result)
{
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

