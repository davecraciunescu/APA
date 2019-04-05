#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
	int nDevices;
	// As the function that points also retrieves the cudaErrorNoDevice
	// the result must be checked
	cudaError_t err = cudaGetDeviceCount(&nDevices);
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
	// Retrives information from every device obtained (if any)
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;

		cudaGetDeviceProperties(&prop, i);

		printf("Device Number: %d\n", i);
		printf("    Device Name: %s\n", prop.name);
		printf("    Memory Clock Rate (KHz): %d\n",
			   prop.memoryClockRate);
		printf("    Memory Bus Width (bits): %d\n",
			   prop.memoryBusWidth);
		printf("    Peak Memory Bandwidth (GB/s): %f\n\n",
			// It is obtained at first bits/s, which wants to be transformed into
			// GB/s -> 1/8 and 1/1000000
			   2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}

	return 0;

}
