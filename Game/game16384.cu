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
// Calloc, exit, free, rand
#include <stdlib.h>
// Time
#include <time.h>

// Size of the tile to be used
#define TILE_WIDTH 1

// Number which allows to know if the matrix is fully occupied
int cellsOccupied = 0;

// -----------------------------------------------------------------------------
// ------------------------------- HEADERS -------------------------------------
// -----------------------------------------------------------------------------
cudaError_t sendMatrixToGpu(char movement, int row, int column, int* matrix);

// -----------------------------------------------------------------------------
// ------------------------------- KERNELS -------------------------------------
// -----------------------------------------------------------------------------
// TODO
/*__global__ computeMatrixUp(int rows, int columns, int* matrix)
{
    // Matrix dimensions
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Location in matrix
    int row =    by * TILE_WIDTH + ty;
    int column = bx * TILE_WIDTH + tx;

    if(row < rows && column < columns)
    {
        
    }

}
*/

/*
 * Fills the empty spaces in the matrix
 *
 * matrix: The matrix which is going to be filled
 * movement: Movement performed by the user (w, s, a, d)
 */
__global__ void fillSpace(int* matrix, char movement, int rows, int columns)
{
    // Matrix dimensions
    int bx = blockIdx.x;  
    int tx = threadIdx.x; 

    // Location in matrix
    int pos = bx * TILE_WIDTH + tx;
    
    switch(movement)
    {
        // Up
        case 'w':
            // From last row to first row
            for(int i = rows - 1; i > 0; i--)
            {
                for(int j = rows - 1; j > 0; j--)
                {
                    // Current cell NOT 0 and upper cell IS 0 ->
                    // moves current cell up
                    if(matrix[j *     rows + pos] != 0 &&
                       matrix[(j - 1) * rows + pos] == 0)
                    {
                        matrix[(j - 1) * rows + pos] = matrix[j * rows + pos];
                        matrix[j * rows + pos] = 0;
                    }
                }
            }
            break;

        // Down
        case 's':
            // From first row to last row
            for(int i = 0; i < rows; i++)
            {
                for(int j = 0; j < rows - 1; j++)
                {
                    // Current cell NOT 0 and lower cell IS 0 ->
                    // moves current cell down
                    if(matrix[j *     rows + pos] != 0 &&
                       matrix[(j + 1) * rows + pos] == 0)
                    {
                        matrix[(j + 1) * rows + pos] = matrix[j * rows + pos];
                        matrix[j * rows + pos] = 0;
                    }
                }
            }
            break;

        // Left
        case 'a':
            // From last row to first row
            for(int i = columns - 1; i > 0; i--)
            {
                for(int j = columns - 1; j > 0; j--)
                {
                    // Current cell NOT 0 and upper cell IS 0 ->
                    // moves current cell up
                    if(matrix[pos * rows +  j]      != 0 &&
                       matrix[pos * rows + (j - 1)] == 0)
                    {
                        matrix[pos * rows + (j - 1)] = matrix[pos * rows + j];
                        matrix[pos * rows +  j] = 0;
                    }
                }
            }
            
            break;

        //  Right
        case 'd':
            // From first row to last row
            for(int i = 0; i < columns; i++)
            {
                for(int j = 0; j < columns - 1; j++)
                {
                    // Current cell NOT 0 and lower cell IS 0 ->
                    // moves current cell down
                    if(matrix[pos * rows + j]       != 0 &&
                       matrix[pos * rows + (j + 1)] == 0)
                    {
                        matrix[pos * rows + (j + 1)] = matrix[pos * rows + j];
                        matrix[pos * rows +  j] = 0;
                    }
                }
            }
            break;
    }

}

// -----------------------------------------------------------------------------
// -------------------------- CUDA RELATED METHODS -----------------------------
// -----------------------------------------------------------------------------
// Method which allows to check for errors
__host__ void check_CUDA_Error(const char *msg)
{
    cudaError_t  err;

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        std::cerr << "ERROR " << err << " OCURRED: " << cudaGetErrorString(err)  
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

    if(nDevices > 0)
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
__host__ void displayGrid(int rows, int columns, int* Matrix)
{
    system("clear");

    /*std::cout << 
        "--------------------------------------------------------------------------------"
        << std::endl;
    std::cout << "16384" << std::endl;    
    std::cout << 
        "--------------------------------------------------------------------------------"
        << std::endl << std::endl;
    */
    // Two extra iterations to print the upper part of the matrix
    for(int i = -2; i < rows; i++)
    {
        if(i < 0) {
            std::cout << "      ";
        } else if(i + 1 < 10) {
            std::cout << i + 1 << " - | ";
        } else if(i + 1 >= 10) {
            std::cout << i + 1 << "- | ";
        }

        for(int j = 0; j < columns; j++)
        {
            if(i == -2) {
                if(j + 1 < 10) {
                    std::cout << j + 1 << "   ";
                } else {
                    std::cout << j + 1 << "  ";
                }
            } else if(i == -1) {
                std::cout << "|   "; 
            } else  {
                std::cout << Matrix[i * rows + j] << " | ";
            }
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout <<                          std::endl 
              <<                          std::endl 
              << "Controls: "          << std::endl
              << "        ___"         << std::endl
              << "       | W |"        << std::endl
              << " ___    ___    ___"  << std::endl
              << "| A |  | S |  | D |" << std::endl
              <<                          std::endl;
}

__host__ void seeding(int gameDifficulty, int rows, int columns, int* matrix)
{
    // Number of seeds to be planted in the board
    int seeds;
    // Values the seeds might have while inserted
    int* seedsValues;
    // Number of seeds planted 
    int seedsPlanted = 0;
    // Position from the matrix where the seed is going to be planted. 
    // Used auxiliary variable
    int position;

    // Depending on the game difficulty, the number of seeds may vary
    switch(gameDifficulty)
    {
        case 1:
            seeds = 15;
            seedsValues = (int*) calloc(3, sizeof(int));
            seedsValues[0] = 2;
            seedsValues[1] = 4;
            seedsValues[2] = 8;
            break;

        case 2:
            seeds = 8;
            seedsValues = (int*) calloc(2, sizeof(int));
            seedsValues[0] = 2;
            seedsValues[1] = 4;
            break;
    }

    // Initialize random seed
    std::srand(time(0));

    while(seedsPlanted < seeds)
    {
        // Still empty cells
        if(cellsOccupied < (rows * columns - 1))
        {
        // Position within the matrix
        position = rand() % ((rows * columns) - 1);

        if(matrix[position] == 0)
        {
                // Random seed value among the ones according to the difficulty
                matrix[position] = seedsValues[rand() % 
                                               (sizeof(seedsValues) 
                                                / sizeof(int) 
                                                - 1)];
                seedsPlanted++;
            }
        } 
        else
        {
            std::cout << "There are no more empty cells, you have lost a live"  << std::endl;
            // TODO:
            // Probably it would be needed decrease the number of lives
            // Therefore, it is needed a new method for that, called here
        }

    }

}

// -----------------------------------------------------------------------------
// -------------------------------- MAIN CODE ----------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Game Mode
    char mode;
    // Game Difficulty
    int  difficulty;
    // Board Height
    int  numRows;
    // Board Weight
    int  numColumns;
    // Board Maximum Number of Cells
    int  numMaxThreads;  
    // Used as auxiliary variable for any input in the system
    std::string input;

    system("clear");
    std::cout << "Processing game settings" << std::endl;

    if(argc == 5)
    {
        input = argv[1];
        
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
            mode = 'm';
        } else if (input.compare("a"))
        {
            std::cout << "Setting to AUTOMATIC" << std::endl;
            std::cout << "-----------------------------" 
                      << " AUTOMATIC MODE COMPLETED 100% " 
                      << "------------------------------"
                      << std::endl;
            mode = 'a';
        } else 
        {
            std::cout <<"There was an error while parsing the input."
                      << std::endl << input 
                      << " is not a valid mode." 
                      << std::endl
                      << "Try writing:\nm: manual\na: automatic"
                      << std::endl;
            exit(0);
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
            difficulty = 1;
        } else if (input.compare("2"))
        {

            std::cout << "Setting to HARD" << std::endl;   
            std::cout << "--------------------------------" 
                      << " HARD MODE COMPLETED 100% " 
                      << "--------------------------------"
                      << std::endl;
            difficulty = 2;
        } else 
        {
            std::cout << "There was an error while parsing the input." 
                      << std::endl << input 
                      << " is not a valid playing dificulty." 
                      << std::endl
                      << "Try writing:\n1: easy\n2: hard" 
                      << std::endl;
            exit(0);
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
            exit(0);
        } 
    
        int* Matrix = (int*) calloc(numRows * numColumns, sizeof(int));
     
        bool play;

        seeding(difficulty, numRows, numColumns, Matrix);
        displayGrid(numRows, numColumns, Matrix);
        
        do {
            std::cin >> input;

            if(input.length() == 1) {
                sendMatrixToGpu(input[0], numRows, numColumns, Matrix);
                displayGrid(numRows, numColumns, Matrix);
            } else {
                std::cout << "not that one cracker!" << std::endl;
            }
        } while(play);

    }
}

// -----------------------------------------------------------------------------
// ----------------------------- CUDA METHODS-----------------------------------
// -----------------------------------------------------------------------------
cudaError_t sendMatrixToGpu(char movement, int row, int column, int* matrix)
{
    // Variable to operate in the GPU
    int* dev_matrix = 0;
    
    // TODO: Set Kernel dimensions correctly, probably need a TILE_WIDTH
    // GPU threads distribution
    dim3 dimGrid(row, column, 1);
    dim3 dimBlock(1, 1);

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
    
    /*
     * If the movement is UP or DOWN:
     *     The number of threads is the number of columns.
     * If the movement is LEFT or RIGHT 
     *     The number of threads is the number fo rows.
     */
    if(movement == 'w' || movement == 's')
    {
        fillSpace<<<1, column>>>(dev_matrix, movement, row, column);
    } 
    else if(movement == 'a' || movement == 'd')
    {
        fillSpace<<<1, row>>>(dev_matrix, movement, row, column);
    }
    check_CUDA_Error("Error while gathering cells\n");

    // Waits for kernel to finish
    cudaDeviceSynchronize();
    check_CUDA_Error("cudaDeviceSynchronize returned error!");

    // Computes the matrix joining the numbers with the same values
    //computeMatrixUp<<<, >>>(rows, columns, dev_matrix);   
    //check_CUDA_Error("Error after trying to mix cells!\n");

    // Waits for kernel to finish
    cudaDeviceSynchronize();
    check_CUDA_Error("cudaDeviceSynchronize returned error!");

    // Memory Transfer: GPU -> CPU
    cudaMemcpy(matrix, dev_matrix, row * column * sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed after copying from GPU to CPU!\n");

    return cudaGetLastError();
}

