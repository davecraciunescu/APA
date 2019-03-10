// -----------------------------------------------------------------------------
// ------------------------------ LIBRARIES ------------------------------------
// -----------------------------------------------------------------------------
// Includes CUDA
#include <cuda_runtime.h>
// CUDA get characteristics
#include <device_launch_parameters.h>
// Allows to implement all string related methods and variables
#include <string>
// C++ library for I/O 
#include <iostream>
// Calloc, exit, free
#include <stdlib.h>

// -----------------------------------------------------------------------------
// ------------------------------- HEADERS -------------------------------------
// -----------------------------------------------------------------------------
cudaError_t sendMatrixToGpu(int row, int column, int* matrix);

// Method which allows to check for errors
__host__ void check_CUDA_Error(const char *msg)
{
    cudaError_t  err;

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        std::cerr << "ERROR " << err << "OCURRED: " << cudaGetErrorString(err)  
                  << "(" << msg << ")" << std::endl;
        std::cerr << "Press any key to finish execution..." << std::endl;
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

    if(cnDevices > 0)
    {
        cudaDeviceProp prop;
        
        cudaGetDeviceProperties(&prop, 0);
        
        threadsBlock = prop.maxThreadsPerBlock;
    }

    return threadsBlock;
}

// -----------------------------------------------------------------------------
// ------------------------------ GAME METHODS ---------------------------------
// -----------------------------------------------------------------------------
void displayGrid(int rows, int columns, int* Matrix)
{
    system("clear");
   
    std::cout << std::endl;

    for(int i = 0; i < rows; i++)
    {
        std::cout << "| ";

        for(int j = 0; j < columns; j++)
        {
            std::cout << Matrix[i * rows + j] << " | ";
        }

        std::cout << std::endl << std::endl;
    }

    std::cout <<                          std::endl 
              <<                          std::endl 
              << "Controls: "          << std::endl
              << "         _"          << std::endl
              << "       | W |"        << std::endl
              << "  _      _      _"   << std::endl
              << "| A |  | S |  | D |" << std::endl
              <<                          std::endl;
}


// -----------------------------------------------------------------------------
// -------------------------------- MAIN CODE ----------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Game Mode
    std::string mode;
    // Game Difficulty
    std::string dificulty;
    // Board Height
    int         numRows;
    // Board Weight
    int         numColumns;
    // Board Maximum Number of Cells
    int         numMaxThreads;  
    
    system("clear");
    std::cout << "Processing game settings" << std::endl;

    if(argc == 5)
    {
        std::string input = argv[1];
        
        // --------------------------------------------------------------------
        // --------------------------- PLAYING MODE ---------------------------
        // --------------------------------------------------------------------
        if(input.compare("m")) 
        {
            std::cout << "Setting to MANUAL" << std::endl;   
            std::cout << "-------------------------------" 
                      << " MANUAL MODE COMPLETED 100% " 
                      << "-------------------------------"
                      << std::endl;
        } else if (input.compare("a"))
        {
            std::cout << "Setting to AUTOMATIC" << std::endl;
            std::cout << "-----------------------------" 
                      << " AUTOMATIC MODE COMPLETED 100% " 
                      << "------------------------------"
                      << std::endl;
        } else 
        {
            std::cout <<"There was an error while parsing the input."
                      << std::endl << input 
                      << " is not a valid mode." 
                      << std::endl
                      << "Try writing:\nm: manual\na: automatic"
                      << std::endl;
        }

        input = argv[2];
        // --------------------------------------------------------------------
        // ------------------------ PLAYING DIFFICULTY ------------------------
        // --------------------------------------------------------------------
        if(input.compare("1")) 
        {
            std::cout << "Setting to EASY" << std::endl;   
            std::cout << "--------------------------------" 
                      << " EASY MODE COMPLETED 100% " 
                      << "--------------------------------"
                      << std::endl;
            
        } else if (input.compare("2"))
        {

            std::cout << "Setting to HARD" << std::endl;   
            std::cout << "--------------------------------" 
                      << " HARD MODE COMPLETED 100% " 
                      << "--------------------------------"
                      << std::endl;
        } else 
        {
            std::cout << "There was an error while parsing the input." 
                      << std::endl << input 
                      << " is not a valid playing dificulty." 
                      << std::endl
                      << "Try writing:\n1: easy\n2: hard" 
                      << std::endl;
        }

        // -------------------------------------------------------------------
        // ------------------------ ROWS & COLUMNS ---------------------------
        // -------------------------------------------------------------------
        numRows    = std::stoi(argv[3]);
        numColumns = std::stoi(argv[4]);

        if(numRows < 0)
        {
            std::cout <<"The number of rows is not valid, it should be bigger than 0"
                      << std::endl;
        }

        if(numColumns < 0)
        {
            std::cout << "The number of columns is not valid, it should be bigger than 0"
                      << std::endl;
        }

        numMaxThreads = getThreadsBlock();
        
        if(numRows * numColumns > numMaxThreads)
        {
            std::cout << "The board is too big, please specify other board size."
                      << "The maximum number of cells is " << numMaxThreads
                      << std::endl;
        }
    
        int* Matrix = (int*) calloc(numRows * numColumns, sizeof(int));

        displayGrid(numRows, numColumns, Matrix);        

    }
}

// -----------------------------------------------------------------------------
// ----------------------------- CUDA METHODS-----------------------------------
// -----------------------------------------------------------------------------
sendMatrixToGpu(int row, int column, int* matrix)
{
    // Variable to operate in the GPU
    int* dev_matrix = 0;
    
    // TODO: Set Kernel dimensions
    // GPU threads distribution
    dim3 dimGrid();
    dim3 dimBlock();

    // Selection of the GPU were code is to be executed
    cudaSetDevice(0);
    check_CUDA_Error("cudaSetDevice failed!\n");
    
    // Memory Allocation for GPU Variables
    cudaMalloc((void**) &dev_matrix, row * column * sizeof(int));
    check_CUDA_Error("cudaMalloc failed at Matrix!\n");

    // Memory Transfer: CPU -> GPU
    cudaMemcpy(dev_matrix, matrix, row * column * sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at Matrix!\n");

    // TODO: Description
    // TODO: Kernel

    // TODO: Insert appropiate failed text for kernel
    check_CUDA_Error("");


    // Waits for kernel to finish
    cudaDeviceSynchronize();
    check_CUDA_Error("cudaDeviceSynchronize returned error!");

    // Memory Transfer: GPU -> CPU
    cudaMemcpy(matrix, dev_matrix, row * column * sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed after copying from GPU to CPU!\n");

    return cudaGetLastError();
}



