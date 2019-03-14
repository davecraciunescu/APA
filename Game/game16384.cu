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
// Rand
#include <cstdlib>
// Time
#include <time.h>
// Slepp
#include <chrono>
#include <thread>

// Size of the tile to be used
#define TILE_WIDTH 1

// -----------------------------------------------------------------------------
// ------------------------------- HEADERS -------------------------------------
// -----------------------------------------------------------------------------
cudaError_t cellsMerge(char movement, int row, int column, int* matrix,
                       int* POINTS, int* CELLS_OCCUPIED);

// -----------------------------------------------------------------------------
// ------------------------------- KERNELS -------------------------------------
// -----------------------------------------------------------------------------
__global__ void computeMatrixUp(int numRows, int numColumns, int* matrix,
                                int* POINTS, int* CELLS_OCCUPIED)
{
    // Matrix dimensions
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Location in matrix
    int col = bx * TILE_WIDTH + tx;

    if(col < numColumns)
    {
        // TODO: Probably will need to change the numRows if TILES or shared 
        // used
        for(int i = 0; i < numRows - 1; i++)
        {
            if(matrix[i * numRows + col] > 0 && 
               matrix[i * numRows + col] == matrix[(i + 1) * numRows + col])
            {
                matrix[i * numRows + col] *= 2;
                matrix[(i + 1) * numRows + col] = 0;
                (*POINTS) += matrix[i * numRows + col];
                (*CELLS_OCCUPIED)--;
            }
        }
    }
}

__global__ void computeMatrixDown(int numRows, int numColumns, int* matrix,
                                  int* POINTS, int* CELLS_OCCUPIED)
{
    // Matrix dimensions
    int bx = blockIdx.x;  
    int tx = threadIdx.x;

    // Location in matrix
    int col = bx * TILE_WIDTH + tx;

    if(col < numColumns)
    {
        // TODO: Probably will need to change the numRows if TILES or shared 
        // used
        for(int i = numRows - 1; i > 0; i--)
        {
            if(matrix[i * numRows + col] > 0 && 
               matrix[i * numRows + col] == matrix[(i - 1) * numRows + col])
            {
                matrix[i * numRows + col] *= 2;
                matrix[(i - 1) * numRows + col] = 0;
                (*POINTS) += matrix[i * numRows + col];
                (*CELLS_OCCUPIED)--;
            }
        }
    }
}

__global__ void computeMatrixLeft(int numRows, int numColumns, int* matrix,
                                  int* POINTS, int* CELLS_OCCUPIED)
{
    // Matrix dimensions
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Location in matrix
    int row = bx * TILE_WIDTH + tx;

    if(row < numRows)
    {
        // TODO: Probably will need to change the numRows if TILES or shared 
        // used
        for(int i = 0; i < numColumns - 1; i++)
        {
            if(matrix[row * numRows + i] > 0 && 
               matrix[row * numRows + i] == matrix[row * numRows + (i + 1)])
            {
                matrix[row * numRows + i] *= 2;
                matrix[row * numRows + (i + 1)] = 0;
                (*POINTS) += matrix[row * numRows + i];
                (*CELLS_OCCUPIED)--;
            }
        }
    }
}

__global__ void computeMatrixRight(int numRows, int numColumns, int* matrix,
                                   int* POINTS, int* CELLS_OCCUPIED)
{
    // Matrix dimensions
    int bx = blockIdx.x;  
    int tx = threadIdx.x;

    // Location in matrix
    int row = bx * TILE_WIDTH + tx;

    if(row < numRows)
    {
        // TODO: Probably will need to change the numRows if TILES or shared 
        // used
        for(int i = numColumns - 1; i > 0; i--)
        {
            if(matrix[row * numRows + i] > 0 && 
               matrix[row * numRows + i] == matrix[row * numRows + (i - 1)])
            {
                matrix[row * numRows + i] *= 2;
                matrix[row * numRows + (i - 1)] = 0;
                (*POINTS) += matrix[row * numRows + i];
                (*CELLS_OCCUPIED)--;
            }
        }
    }
}

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
// ------------------------- FUNCTIONALITY METHODS  ----------------------------
// -----------------------------------------------------------------------------
// Method which allows to check for errors
__host__ void check_CUDA_Error(const char *msg)
{
    cudaError_t  err;

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        std::cout << "ERROR " << err << " OCURRED: " << cudaGetErrorString(err)  
                  << "(" << msg << ")" << std::endl;
        std::cout << "Press any key to finish execution..." << std::endl;
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
__host__ std::string printHearts(int* LIVES)
{
    std::string hearts;

    for(int i = 0; i < *LIVES; i++) {
        hearts += "<3 ";
    }

    return hearts;
}

__host__ void displayGrid(int rows, int columns, int* Matrix, 
                          int* POINTS, int* LIVES, int* CELLS_OCCUPIED)
{
    system("clear");

    // Two extra iterations to print the upper part of the matrix
    for(int i = -2; i < rows; i++)
    {
        if(i < 0) {
            std::cout << "      ";
        } else if(i + 1 < 10) {
            std::cout << i + 1 << " - ";
        } else if(i + 1 >= 10) {
            std::cout << i + 1 << "- ";
        }

        for(int j = 0; j < columns; j++)
        {
            if(i == -2) {
                if(j + 1 < 10) {
                    std::cout << j + 1 << "    ";
                } else {
                    std::cout << j + 1 << "   ";
                }
            } else if(i == -1) {
                std::cout << "|    "; 
            } else  {

                switch(Matrix[i * rows + j])
                {
                    // LIGHTWHITE
                    case 2:
                        std::cout << "\033[1;37;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;
 
                    // WHITE
                    case 4:
                        std::cout << "\033[1;37m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;
                    
                    // DARKGRAY
                    case 8:
                        std::cout << "\033[1;30;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;
                    
                    // YELLOW
                    case 16:
                        std::cout << "\033[1;33m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // LIGHTMAGENTA
                    case 32:
                        std::cout << "\033[1;35;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;
                    
                    // MAGENTA
                    case 64:
                        std::cout << "\033[1;35m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // LIGHTRED
                    case 128:
                        std::cout << "\033[1;31;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // RED
                    case 256:
                        std::cout << "\033[1;31m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // TODO
                    // BROWN
                    case 512:
                        std::cout << "\033[1;37m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // LIGHTGREEN
                    case 1024:
                        std::cout << "\033[1;32;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // GREEN
                    case 2048:
                        std::cout << "\033[1;32m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;
                    
                    // LIGHTCYAN
                    case 4096:
                        std::cout << "\033[1;36;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // CYAN
                    case 8192:
                        std::cout << "\033[1;36m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    // LIGHTBLUE
                    case 16384:
                        std::cout << "\033[1;34;1m| " << Matrix[i * rows + j] 
                                  << " |\033[0m";
                        break;

                    default:
                        std::cout << "| " << Matrix[i * rows + j] << " |";
                        
                        break;
                }
            }
        }

        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl << std::endl 
              << "Controls:            Save    Quit     Points:"
              << "        Cells           Lives:"                   << std::endl
              << "        ___                           "
              << "               Occupied:"                         << std::endl
              << "       | W |          ___     ___     "           << std::endl
              << " ___    ___    ___   | G |   | Q |    "           << *POINTS 
              << "              " << *CELLS_OCCUPIED                << std::endl
              << "| A |  | S |  | D |                                          "
              << "        " << printHearts(LIVES) 
              << std::endl << std::endl;
}

__host__ void seeding(int gameDifficulty, int rows, int columns, int* matrix,
                      int* CELLS_OCCUPIED)
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
    std::srand(time(NULL));

    std::cout << "CELLS_OCCUPIED: " << (*CELLS_OCCUPIED) << std::endl;

    while(seedsPlanted < seeds)
    {
        // Still empty cells
        if((*CELLS_OCCUPIED) <= (rows * columns))
        {
            // Position within the matrix
            position = rand() % (rows * columns);

            if(matrix[position] == 0)
            {
                // Random seed value among the ones according to the difficulty
                matrix[position] = seedsValues[rand() % 
                                               (sizeof(seedsValues) 
                                                / sizeof(int))];
                seedsPlanted++;
                (*CELLS_OCCUPIED)++;     
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

/**
* Asks user if will play again.
*/
bool playAgain(int lives)
{
    bool willPlayAgain = false;

    std::cout << "You currently have: " << lives << " lives." << std::endl;
    std::cout << "Do you want to play again (y/n).";

    std::string input;
    std::cin >> input;

    bool invalid = true;

    while (invalid)
    {
        if (input.length() == 1)
        {
            switch(input[0])
            {
                case 'y':
                    std::cout << "Alright, playing again." << std::endl;
                    willPlayAgain   = true;
                    invalid         = false;
                    break;
                
                case 'n':
                    std::cout << "Thanks for playing." << std::endl;
                    invalid         = false;
                    break;
                
                default:
                    std::cout << "Please enter a valid value." << std::endl;
                    std::cout << "Do you want to play again (y/n).";
                    break;
            }
        }
        else
        {
            std::cout << "Please enter a valid value." << std::endl;
        }
    }
    return willPlayAgain;
}

void playGameManual (
    int  difficulty,    // Difficulty of the game.
    int  numRows,       // Number of rows in the game.
    int  numColumns,    // Number of columns in the game.
    int  numMaxThreads  // Number of max threads to be run.
    )
{

    // Auxiliary input variable.
    std::string input;

    // Variables needed within the game.
    int   lives = 5;    int* LIVES          = &lives;
    int  points = 0;    int* POINTS         = &points;
    int cellsOc = 0;    int* CELLS_OCCUPIED = &cellsOc; 

    int*       matrix = (int*) calloc(numRows * numColumns, sizeof(int)); 
    bool      playing = true;
    bool  keepPlaying = true;
    bool       winner = true;

    // MAIN GAME LOOP.
    while (keepPlaying)
    {   
        // SEED GAME INITIAL STATE.
        seeding(difficulty, numRows, numColumns, matrix, CELLS_OCCUPIED);
        displayGrid(numRows, numColumns, matrix, POINTS, LIVES, CELLS_OCCUPIED);

        std::cout << "Starting Game." << std::endl;

        if (lives > 0)
        {
            while (playing)
            {
                std::cin >> input;

                if (input.length() == 1)
                {
                    switch (input[0])
                    {
                        case 'g':
                            // TODO SAVE GAME.
                        break;
                        
                        case 'q':
                            playing        = false;
                            keepPlaying    = false;
                            winner         = false;
                        break;
                        
                        default:
                            if (winner) // TODO: CONNECT WINNER WITH KERNEL.
                            {
                                cellsMerge(input[0], numRows, numColumns, matrix,
                                    POINTS, CELLS_OCCUPIED);
                                seeding(difficulty, numRows, numColumns, matrix,
                                    CELLS_OCCUPIED);
                                displayGrid(numRows, numColumns, matrix, POINTS, LIVES,
                                    CELLS_OCCUPIED);
                            }
                            else
                            {
                                // Take away one life.
                                lives--;
                                playing = false;
                            }
                        break;
                    }
                }
                else
                {
                    std:: cout << "Not that one, cracker!" << std::endl;
                }
            }

            // ASK USER IF WANTS TO PLAY AGAIN.
            keepPlaying = playAgain(lives);
        }
        else
        {
            std::cout << "You have 0 lives. GAMEOVER." << std::endl;
            keepPlaying = false;
        }
    }

    // Reset the value of winner.
    winner = true;
}

void playGameAutomatic (
    int  difficulty,    // Difficulty of the game.
    int  numRows,       // Number of rows in the game.
    int  numColumns,    // Number of columns in the game.
    int  numMaxThreads  // Number of max threads to be run.
    )
{

    // Auxiliary input variable.
    std::string input;

    // Variables needed within the game.
    int   lives = 5;    int* LIVES          = &lives;
    int  points = 0;    int* POINTS         = &points;
    int cellsOc = 0;    int* CELLS_OCCUPIED = &cellsOc; 

    int*       matrix = (int*) calloc(numRows * numColumns, sizeof(int)); 
    bool      playing = true;
    bool  keepPlaying = true;
    bool       winner = true;

    char movements [4] = {'w', 'a', 's', 'd'};

    // Initialize random seed
    std::srand(time(NULL));

    /* Used to know when to ask the user for an action:
     * - Continue.
     * - Quit.
     * - Save.
     */
    int iteration = 0;

    // MAIN GAME LOOP.
    while (keepPlaying)
    {   
        // SEED GAME INITIAL STATE.
        seeding(difficulty, numRows, numColumns, matrix, CELLS_OCCUPIED);
        displayGrid(numRows, numColumns, matrix, POINTS, LIVES, CELLS_OCCUPIED);

        std::cout << "Starting Game." << std::endl;

        if (lives > 0)
        {
            while (playing)
            {   

                std::this_thread::sleep_for(std::chrono::milliseconds(1200));

                if(iteration % 10 == 0)
                {
                    std::cout << "Do you wish to SAVE (G) your game? "
                              << std::endl
                              << "Or maybe to QUIT (Q) the game?"
                              << std::endl
                              << "If you want to keep playing in the "
                              << "automatical mode PRESS ANY KEY."
                              << std::endl;

                    std::cin >> input;
                    iteration = 1;
                    
                    if (input.length() == 1)
                    {
                        switch (input[0])
                        {
                            case 'g':
                                // TODO SAVE GAME.
                            break;
                            
                            case 'q':
                                playing        = false;
                                keepPlaying    = false;
                                winner         = false;
                            break;
                            
                            default:
                                if (winner) // TODO: CONNECT WINNER WITH KERNEL.
                                {
                                    cellsMerge(input[0], numRows, numColumns, matrix,
                                        POINTS, CELLS_OCCUPIED);
                                    seeding(difficulty, numRows, numColumns, matrix,
                                        CELLS_OCCUPIED);
                                    displayGrid(numRows, numColumns, matrix, POINTS, LIVES,
                                        CELLS_OCCUPIED);
                                }
                                else
                                {
                                    // Take away one life.
                                    lives--;
                                    playing = false;
                                }
                            break;
                        }
                    }
                    else
                    {
                        std:: cout << "Not that one, cracker!" << std::endl;
                    }
                } 
                else 
                {
                    iteration++;

                    if (winner) // TODO: CONNECT WINNER WITH KERNEL.
                    {
                        int movement = rand() %
                                       (sizeof(movements) / sizeof(char));
                        cellsMerge(movements[movement], numRows, numColumns, matrix,
                                   POINTS, CELLS_OCCUPIED);
                        seeding(difficulty, numRows, numColumns, matrix,
                                CELLS_OCCUPIED);
                        displayGrid(numRows, numColumns, matrix, POINTS, LIVES,
                                    CELLS_OCCUPIED);
                    }
                    else
                    {
                        // Take away one life.
                        lives--;
                        playing = false;
                    }
                }

            }

            // ASK USER IF WANTS TO PLAY AGAIN.
            keepPlaying = playAgain(lives);
        }
        else
        {
            std::cout << "You have 0 lives. GAMEOVER." << std::endl;
            keepPlaying = false;
        }
    }

    // Reset the value of winner.
    winner = true;
    
}

// MAIN METHOD OF THE GAME.
void playGame (
    int  difficulty,    // Difficulty of the game.
    int  numRows,       // Number of rows in the game.
    int  numColumns,    // Number of columns in the game.
    int  numMaxThreads, // Number of max threads to be run.
    char mode           // Gaming mode (manual or automatic).
    )
{
    switch(mode)
    {
        case 'm':
            playGameManual(difficulty, numRows, numColumns, numMaxThreads);
            break;

        case 'a':
            playGameAutomatic(difficulty, numRows, numColumns, numMaxThreads);
            break;
    }
}

// -----------------------------------------------------------------------------
// -------------------------------- MAIN CODE ----------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Variables needed as game settings.
    char mode;          // Game mode.
    int  difficulty;    // Difficulty of the game.
    int  numRows;       // Number of rows in the game.
    int  numColumns;    // Number of columns in the game.
    int  numMaxThreads; // Number of max threads to be run.
    
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
        if(input.compare("m") == 0) 
        {
            std::cout << "Setting to MANUAL" << std::endl;   
            std::cout << "-------------------------------" 
                      << " MANUAL MODE COMPLETED 100% " 
                      << "-------------------------------"
                      << std::endl;
            mode = 'm';
        } else if (input.compare("a") == 0)
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
        if(input.compare("1") == 0) 
        {
            std::cout << "Setting to EASY" << std::endl;   
            std::cout << "--------------------------------" 
                      << " EASY MODE COMPLETED 100% " 
                      << "--------------------------------"
                      << std::endl;
            difficulty = 1;
        } else if (input.compare("2") == 0)
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
            exit(0);
        }

        if(numColumns < 0)
        {
            std::cout << "The number of columns is not valid, it should be bigger than 0"
                      << std::endl;
            exit(0);
        }

        numMaxThreads = getThreadsBlock();
        
        if(numRows * numColumns > numMaxThreads)
        {
            std::cout << "The board is too big, please specify other board size."
                      << "The maximum number of cells is " << numMaxThreads
                      << std::endl;
            exit(0);
        } 
       
        // EXECUTE GAME.
        playGame(difficulty, numRows, numColumns, numMaxThreads, mode);
    }
}


// -----------------------------------------------------------------------------
// ----------------------------- CUDA METHODS-----------------------------------
// -----------------------------------------------------------------------------
cudaError_t cellsMerge(char movement, int row, int column, int* matrix, 
                       int* POINTS, int* CELLS_OCCUPIED)
{
    // Variable to operate in the GPU
    int* dev_matrix = 0;
    int* dev_POINTS = 0;
    int* dev_CELLSO = 0;

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

    cudaMalloc((void**) &dev_POINTS, sizeof(int));
    check_CUDA_Error("cudaMalloc failed at POINTS!\n");

    cudaMalloc((void**) &dev_CELLSO, sizeof(int));
    check_CUDA_Error("cudaMalloc failed at CELLS_OCCUPIED!\n");

    // Memory Transfer: CPU -> GPU
    cudaMemcpy(dev_matrix, matrix, row * column * sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at Matrix (CPU -> GPU)!\n");
    
    cudaMemcpy(dev_POINTS, POINTS, sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at POINTS (CPU -> GPU)!\n");
    
    cudaMemcpy(dev_CELLSO, CELLS_OCCUPIED, sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at CELLS_OCCUPIED (CPU -> GPU)!\n");
    
    /*
     * If the movement is UP or DOWN:
     *     The number of threads is the number of columns.
     * If the movement is LEFT or RIGHT 
     *     The number of threads is the number fo rows.
     */
    switch(movement)
    {
        case 'w':
            fillSpace<<<1, column>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");

            computeMatrixUp<<<1, column>>>(row, column, dev_matrix, 
                                           dev_POINTS, dev_CELLSO);
            check_CUDA_Error("Error merging cells!\n");
            
            fillSpace<<<1, column>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");
            break;

        case 's':
            fillSpace<<<1, column>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");

            computeMatrixDown<<<1, column>>>(row, column, dev_matrix,
                                             dev_POINTS, dev_CELLSO);
            check_CUDA_Error("Error merging cells!\n");
            
            fillSpace<<<1, column>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");
            break;
        
        case 'a':
            fillSpace<<<1, row>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");

            computeMatrixLeft<<<1, row>>>(row, column, dev_matrix, 
                                          dev_POINTS, dev_CELLSO);
            check_CUDA_Error("Error merging cells!\n");
            
            fillSpace<<<1, row>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");
            break;
        
        case 'd':
            fillSpace<<<1, row>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");

            computeMatrixRight<<<1, row>>>(row, column, dev_matrix,
                                           dev_POINTS, dev_CELLSO);
            check_CUDA_Error("Error merging cells!\n");
            
            fillSpace<<<1, row>>>(dev_matrix, movement, row, column);
            check_CUDA_Error("Error while gathering cells!\n");
            break;
    }
    
    // Waits for kernel to finish
    cudaDeviceSynchronize();
    check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

    // Waits for kernel to finish
    cudaDeviceSynchronize();
    check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

    // Memory Transfer: GPU -> CPU
    cudaMemcpy(matrix, dev_matrix, row * column * sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at Matrix (GPU -> CPU)!\n");

    cudaMemcpy(POINTS, dev_POINTS, sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at POINTS (GPU -> CPU)!\n");

    cudaMemcpy(CELLS_OCCUPIED, dev_CELLSO, sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at CELLS_OCCUPIED (GPU -> CPU)!\n");
    return cudaGetLastError();
}

