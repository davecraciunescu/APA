// Includes CUDA
#include <cuda_runtime.h>
// CUDA get characteristics
#include <device_launch_parameters.h>
#include <.\inc\helper_cuda.h>
// Allows to implement all string related methods and variables
#include <string>

int main(int argc, char** argv)
{
    std::string mode;
    std::string dificulty;
    int         height;
    int         weight;

    if(argc == 4)
    {
        std::string input = argv[0];
        
        // --------------------------------------------------------------------
        // --------------------------- PLAYING MODE ---------------------------
        // --------------------------------------------------------------------
        if(input.compare("m")) 
        {
            
        } else if (input.compare("a"))
        {

        } else 
        {
            printf("There was an error while parsing the input.\n");
            printf("%s is not a valid mode.\n", input);
            printf("Try writing:\nm: manual\na: automatic\n");
        }

        // --------------------------------------------------------------------
        // ------------------------ PLAYING DIFFICULTY ------------------------
        // --------------------------------------------------------------------
        if(input.compare("1")) 
        {
            
        } else if (input.compare("2"))
        {

        } else 
        {
            printf("There was an error while parsing the input.\n");
            printf("%s is not a valid playing dificulty.\n", input);
            printf("Try writing:\n1: easy\n2: hard\n");
        }

        // -------------------------------------------------------------------
        // ----------------------- HEIGHT & WEIGHT --------------------------
        // -------------------------------------------------------------------
        height = std::stoi(argv[2]);
        weight = std::stoi[argv[3]];

        if(height < 0)
        {
            printf("The height number is not valid, it should be bigger than 0");
        }

        if(weight < 0)
        {
            printf("The weight number is not valid, it should be bigger than 0");
        }


    }

}

// Method which allows to check for errors
__host__ void check_CUDA_Error(const char *msg)
{
    cudaError_t  err;

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        fprintf("ERROR %d OCURRED: %s (%s)\n", err, cudaGetErrorString(err), msg);
        fprintf("Press any key to finish execution...");
        fflush(stdin);

        char key = getchar();

        exit(-1);
    }
}

// Gets the number of threads allowed per block in the current GPU
__host__ int getThreadsBlock()
{
    int threadsBlock = 0;

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    check_CUDA_Error("Couldn't get the number of devices in this computer");

    for(int i = 0; i < ; i++)
    {
        cudaDeviceProp prop;
        
        cudaGetDeviceProperties(&prop, i);
        
        threadsBlock = prop.maxThreadsPerBlock;
    }

    return threadsBlock;
}

