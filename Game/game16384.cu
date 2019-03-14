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
#include <fstream>
// Calloc, exit, free
#include <stdlib.h>
// Rand
#include <cstdlib>
// Time
#include <time.h>
// Slepp
#include <chrono>
#include <thread>
// Fill in 
#include <algorithm>

// Size of the tile to be used
#define TILE_WIDTH 2

// -----------------------------------------------------------------------------
// ------------------------------- HEADERS -------------------------------------
// -----------------------------------------------------------------------------
void check_CUDA_Error(const char *msg);

int getThreadsMaxBlock();

int getMinBoard(int difficulty);
    
std::string printHearts(int LIVES);

__host__ void displayGrid(int rows,             // Rows of the table.
                          int columns,          // Columns of the table.
                          int* Matrix,          // Matrix with values.
                          int* POINTS,          // Earned points.
                          int* LIVES,           // Remaining lives
                          int* CELLS_OCCUPIED,  // Occupied cells.
                          int* columnLength);   // Length of the columns.
    
__host__ bool seeding(int gameDifficulty,       // Gaming difficulty.
                      int rows,                 // Rows of the table.
                      int columns,              // Columns of the table.
                      int* matrix,              // Matrix with values.
                      int* CELLS_OCCUPIED);     // Occupied cells.

bool playAgain(int *LIVES);

char randomMovement();

void playGame(
     int difficulty,        // Difficulty of the game.
     int numRows,           // Number of rows.
     int numColumns,        // Number of columns.
     int numMaxThreads,     // Max number of threds.
    int* columnLength,      // Length of the column.
    bool automatic,         // Automatic gamemode.
    int* matrix);           // Matrix with values.

void saveGame(
     int difficulty,        // Difficulty of the game.
     int numRows,           // Number of rows in the game.
     int numColumns,        // Number of columns in the game.
     int numMaxThreads,     // Number of max threads to be run.
    int* columnLength,      // Length of bigger number in column.
    bool automatic,         // Play game in automatic mode.
    int* matrix);           // matrix with values.

void loadGame();

cudaError_t cellsMerge(
    char movement,          // Direction of the movement. 
     int row,               // Rows of the table.
     int column,            // Columns of the table.
    int* matrix,            // Matrix with values.
    int* POINTS,            // Number of points.
    int* CELLS_OCCUPIED,    // Occupied cells.
    int* columnLength);     // Length of the columns.

cudaError_t cellsMerge(
    char movement,          // Direction of the movement.
    int  numRows,           // Rows of the table.
    int  numColumns,        // Columns of the table.
    int* matrix,            // Matrix with values.
    int* POINTS,            // Number of points.
    int* CELLS_OCCUPIED,    // Occupied cells.
    int* columnLength);     // Length of the columns.

// -----------------------------------------------------------------------------
// ------------------------------- KERNELS -------------------------------------
// -----------------------------------------------------------------------------
__global__ void computeMatrixUp(int  numRows, 
                                int  numColumns, 
                                int* matrix,
                                int* POINTS, 
                                int* CELLS_OCCUPIED, 
                                int* columnLength)
{
    // Matrix dimensions
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Location in matrix
    int col = bx * TILE_WIDTH + tx;

    if(col < numColumns)
    {
        for(int i = 0; i < numRows - 1; i++)
        {
            if(matrix[i * numRows + col] > 0 && 
               matrix[i * numRows + col] == matrix[(i + 1) * numRows + col])
            {
                matrix[i * numRows + col] *= 2;
                matrix[(i + 1) * numRows + col] = 0;
      
                atomicAdd(POINTS, matrix[i * numRows + col]);
                atomicAdd(CELLS_OCCUPIED, -1);

                /*
                if(columnLength[col] 
                   < 
                   std::to_string(matrix[i * numRows + col]).length())
                {
                    columnLength[col] = matrix[i * numRows + col];
                }
                */
            }
        }
    }
}

__global__ void computeMatrixDown(int  numRows,
                                  int  numColumns, 
                                  int* matrix,
                                  int* POINTS, 
                                  int* CELLS_OCCUPIED,
                                  int* columnLength)
{
    // Matrix dimensions
    int bx = blockIdx.x;  
    int tx = threadIdx.x;

    // Location in matrix
    int col = bx * TILE_WIDTH + tx;

    if(col < numColumns)
    {
        for(int i = numRows - 1; i > 0; i--)
        {
            if(matrix[i * numRows + col] > 0 && 
               matrix[i * numRows + col] == matrix[(i - 1) * numRows + col])
            {
                matrix[i * numRows + col] *= 2;
                matrix[(i - 1) * numRows + col] = 0;
                
                atomicAdd(POINTS, matrix[i * numRows + col]);
                atomicAdd(CELLS_OCCUPIED, -1);
                
                /* 
                if(columnLength[col] 
                   < 
                   std::to_string(matrix[i * numRows + col]).length())
                {
                    columnLength[col] = matrix[i * numRows + col];
                }
                */
            }
        }
    }
}

__global__ void computeMatrixLeft(int  numRows, 
                                  int  numColumns, 
                                  int* matrix,
                                  int* POINTS, 
                                  int* CELLS_OCCUPIED, 
                                  int* columnLength)
{
    // Matrix dimensions
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Location in matrix
    int row = bx * TILE_WIDTH + tx;

    if(row < numRows)
    {
        for(int i = 0; i < numColumns - 1; i++)
        {
            if(matrix[row * numRows + i] > 0 && 
               matrix[row * numRows + i] == matrix[row * numRows + (i + 1)])
            {
                matrix[row * numRows + i] *= 2;
                matrix[row * numRows + (i + 1)] = 0;
                
                atomicAdd(POINTS, matrix[row * numRows + i]);
                atomicAdd(CELLS_OCCUPIED, -1);
                
                /*
                if(columnLength[i] 
                   < 
                   std::to_string(matrix[row * numRows + i]).length())
                {
                    columnLength[i] = matrix[row * numRows + i];
                }
                */
            }
        }
    }
}

__global__ void computeMatrixRight(int  numRows, 
                                   int  numColumns, 
                                   int* matrix,
                                   int* POINTS, 
                                   int* CELLS_OCCUPIED,
                                   int* columnLength)
{
    // Matrix dimensions
    int bx = blockIdx.x;  
    int tx = threadIdx.x;

    // Location in matrix
    int row = bx * TILE_WIDTH + tx;

    if(row < numRows)
    {
        for(int i = numColumns - 1; i > 0; i--)
        {
            if(matrix[row * numRows + i] > 0 && 
               matrix[row * numRows + i] == matrix[row * numRows + (i - 1)])
            {
                matrix[row * numRows + i] *= 2;
                matrix[row * numRows + (i - 1)] = 0;

                atomicAdd(POINTS, matrix[row * numRows + i]);
                atomicAdd(CELLS_OCCUPIED, -1);
                
                /*
                if(columnLength[i] 
                   < 
                   std::to_string(matrix[row * numRows + i]).length())
                {
                    columnLength[i] = matrix[row * numRows + i];
                }
                */
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
__host__ int getThreadsMaxBlock()
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

// Gets the apropiate sixe for the gaming mode specified
__host__ int getMinBoard(int difficulty)
{
    // In each difficulty level, the player should be allowed to perform at
    // least three movements.
    switch(difficulty)
    {
        case 1:
            return 3 * 15;
        case 2:
            return 3 * 8;
        default:
            return 0;
    }
}

// -----------------------------------------------------------------------------
// ------------------------------ GAME METHODS ---------------------------------
// -----------------------------------------------------------------------------

// Returns as many Pixel Art Herats as specified by parameter.
__host__ std::string printHearts(int LIVES)
{
    std::string hearts;

    for(int i = 0; i < LIVES; i++) {
        hearts += "<3 ";
    }

    return hearts;
}

// Prints the game's grid, including buttons, lives and punctuation.
__host__ void displayGrid(int rows,             // Rows of the table.
                          int columns,          // Columns of the table.
                          int* Matrix,          // Matrix with values.
                          int* POINTS,          // Earned points.
                          int* LIVES,           // Remaining lives
                          int* CELLS_OCCUPIED,  // Occupied cells.
                          int* columnLength)    // Length of the columns.
{
    system("clear");
   
    // Game's title
    std::cout << "                       "
              << "  _    ____     __       __    __ __      " 
              << std::endl;
    std::cout << "                       "
              << "/' \\  /'___\\  /'__`\\   /'_ `\\ /\\ \\\\ \\     " 
              << std::endl;
    std::cout << "                       " 
              << "\\_, \\/\\ \\__/ /\\_\\L\\ \\ /\\ \\L\\ \\\\ \\ \\\\ \\    " 
              << std::endl;
    std::cout << "                       "
              << "/_/\\ \\ \\  _``\\/_/_\\_<_\\/_> _ <_\\ \\ \\\\ \\_  " 
              << std::endl;
    std::cout << "                       "
              << "  \\ \\ \\ \\ \\L\\ \\/\\ \\L\\ \\ /\\ \\L\\ \\\\ \\__ ,__\\" 
              << std::endl;
    std::cout << "                       " 
              << "   \\ \\_\\ \\____/\\ \\____/ \\ \\____/ \\/_/\\_\\_/" 
              << std::endl;
    std::cout << "                       "
              << "    \\/_/\\/___/  \\/___/   \\/___/     \\/_/  " 
              << std::endl << std::endl;

    // Prints the matrix.
    // Two extra iterations to print the upper part of the matrix.
    for(int i = -2; i < rows; i++)
    {
        // Rows IDs.
        if(i < 0) {
            std::cout << "      ";
        } else if(i + 1 < 10) {
            std::cout << i + 1 << " - ";
        } else if(i + 1 >= 10) {
            std::cout << i + 1 << "- ";
        }

        // Rows.
        for(int j = 0; j < columns; j++)
        {
            // Columns IDs.
            if(i == -2) {
                if(j + 1 < 10) {
                    std::cout << j + 1 << "    ";
                } else {
                    std::cout << j + 1 << "   ";
                }
            } else if(i == -1) {
                std::cout << "|    "; 
            } else 
            {
                // Matrix's values printed by colors depending on their value.
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

                    // TODO: At the begging it was used other system for
                    // coloring, but it end up not working. In the current
                    // system used it hasn't been found any BROWN color.
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

                    // NO COLOR
                    default:
                        std::cout << "| " << Matrix[i * rows + j] << " |";
                        
                        break;
                }
            }
        }

        std::cout << std::endl;
    }

    // Controls and options.
    std::cout << std::endl << std::endl << std::endl 
              << "Controls:            Save    Quit     Points:"
              << "        Cells           Lives:"                   << std::endl
              << "        ___                           "
              << "               Occupied:"                         << std::endl
              << "       | W |          ___     ___     "           << std::endl
              << " ___    ___    ___   | G |   | Q |    "           << *POINTS 
              << "              " << *CELLS_OCCUPIED                << std::endl
              << "| A |  | S |  | D |                                          "
              << "        " <<  printHearts(*LIVES) 
              << std::endl << std::endl;
}

// Inserts new seeds into the gaming board randomly.
__host__ bool seeding(int gameDifficulty, 
                      int rows, 
                      int columns, 
                      int* matrix,
                      int* CELLS_OCCUPIED)
{
    // Number of seeds to be planted in the board
    int seeds;
    // Values the seeds might have while inserted
    int* seedsValues;
    // Number of seeds planted 
    int seedsPlanted = 0;
    // Number of different seeds per level of difficulty
    int differentValues;
    // Position from the matrix where the seed is going to be planted. 
    // Used auxiliary variable
    int position;

    bool canPlay = true; // Defines if player has lost the game.

    // Depending on the game difficulty, the number of seeds may vary
    switch(gameDifficulty)
    {
        // EASY
        case 1:
            seeds = 15;
            differentValues = 3;
            seedsValues = (int*) calloc(differentValues, sizeof(int));
            seedsValues[0] = 2;
            seedsValues[1] = 4;
            seedsValues[2] = 8;
            break;

        // HARD
        case 2:
            seeds = 8;
            differentValues = 2;
            seedsValues = (int*) calloc(differentValues, sizeof(int));
            seedsValues[0] = 2;
            seedsValues[1] = 4;
            break;
    }

    // Initialize random seed
    std::srand(time(NULL));

    while(canPlay && seedsPlanted < seeds)
    {
        // Still empty cells
        if((*CELLS_OCCUPIED) < (rows * columns))
        {
            // Position within the matrix
            position = rand() % (rows * columns);

            if(matrix[position] == 0)
            {
                // Random seed value among the ones according to the difficulty
                matrix[position] = seedsValues[rand() % differentValues];
                seedsPlanted++;
                (*CELLS_OCCUPIED)++;     
            }
        } 
        else
        {
            std::cout << "There are no more empty cells, you have lost a life"  << std::endl;
            canPlay = false;
        }

    }
    return canPlay;
}

/**
* Asks user if will play again.
*/
bool playAgain(int *LIVES)
{
    // Allows to know if the user wants to play once he has lost a live.
    bool willPlayAgain;

    std::cout << "You currently have: " << *LIVES << " lives." << std::endl;
    std::cout << "Do you want to play again (y/n).";

    std::string input;

    bool invalid = true;

    // Asks for an input as long as it is not YES or NO the answer given.
    while (invalid)
    {
        std::cin >> input;

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
                    willPlayAgain   = false;
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

// Returns a random movement.
char randomMovement()
{
    char movements [4] = {'w','a','s','d'};

    // Initialize random seeding.
    std::srand(time(NULL));

    return movements[rand() % (sizeof(movements) / sizeof(char))]; 
}

// Playing the game.
void playGame (
    int  difficulty,    // Difficulty of the game.
    int  numRows,       // Number of rows in game.
    int  numColumns,    // Number of columns in game.
    int  numMaxThreads, // Number of max threads to be run.
    int* columnLength,  // Length of bigger number in column.
    bool automatic,     // Play game in automatic mode.
    int* matrix)        // Matrix with values.
{
    // Auxiliary input variable.
    std::string input;

    // Variables needed within the game.
    int   lives = 5;    int* LIVES          = &lives;
    int  points = 0;    int* POINTS         = &points;
    int cellsOc = 0;    int* CELLS_OCCUPIED = &cellsOc;

    bool     playing = true;
    bool keepPlaying = true;
     int   iteration = 0;

    // Main Game Loop.
    while (keepPlaying)
    {
        std::cout << "Starting Game." << std::endl;

        if (lives > 0)
        {
            // Seed game to initial State.
            seeding(difficulty, numRows, numColumns, matrix, CELLS_OCCUPIED);
            displayGrid(numRows, numColumns, matrix, POINTS, LIVES,
                CELLS_OCCUPIED, columnLength);

            while (playing)
            {
                // Gameplay changes if gamemode is automatic.
                if (automatic)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                    if (iteration != 10)
                    {
                        iteration++;
                        input = randomMovement(); // Autogenerate a move.
                    }
                    else
                    { 
                        iteration = 1;

                        std::cout << "Do you with to SAVE (G) your game?"
                                  << std::endl
                                  << "Or maybe to QUIT (Q) the game?"
                                  << std::endl
                                  << "If you want to keep playing in the "
                                  << "automatic mode PRESS ANY KEY."
                                  << std::endl;
                        
                        // Get user input.
                        std::cin >> input;
                    }
                }
                else
                {
                    std::cin >> input;
                }

                // Treat input as if the same behaviour. Automode has been
                // considered

                if (input.length() == 1)
                {
                    switch(input[0])
                    {
                        case 'g':
                            saveGame(difficulty, numRows, numColumns,
                            numMaxThreads, columnLength, automatic, matrix);
                        break;

                        case 'q':
                            playing     = false;
                            keepPlaying = false;
                        break;

                        default:
                            cellsMerge(input[0], numRows, numColumns, matrix,
                                POINTS, CELLS_OCCUPIED, columnLength);

                            // Check if board is full.
                            if (seeding(difficulty, numRows, numColumns, matrix,
                                CELLS_OCCUPIED))
                            {
                                displayGrid(numRows, numColumns, matrix, POINTS,
                                    LIVES, CELLS_OCCUPIED, columnLength);
                            }
                            else
                            {
                                lives--; // Take away one life.
                                *CELLS_OCCUPIED = 0;
                                free(matrix);
                                matrix = (int*) calloc(numRows * numColumns,
                                                       sizeof(int));
                                playing = false;
                            }
                        break;
                    }
                }
                else
                {
                    std::cout << "Not that one, cracker!" << std::endl;
                }
            }

            playing = true;

            // Ask if user wants to play again.
            keepPlaying = playAgain(LIVES);
        }
        else
        {
            std::cout << "You have 0 lives. GAMEOVER." << std::endl;
            keepPlaying = false;
        }
    }
}

// Saves the current status of the game and all its settings.
void saveGame(
     int difficulty,        // Difficulty of the game.
     int numRows,           // Number of rows in the game.
     int numColumns,        // Number of columns in the game.
     int numMaxThreads,     // Number of max threads to be run.
    int* columnLength,      // Length of bigger number in column.
    bool automatic,         // Play game in automatic mode.
    int* matrix)            // Matrix with the values.
{
    // File exportation variables.
    std::ofstream file;

    // Parse boolean value.
    std::string boolean = automatic ? "true":"false";
   
    file.open("save.txt");

    file << std::to_string(difficulty)    << std::endl;
    file << std::to_string(numRows)       << std::endl;
    file << std::to_string(numColumns)    << std::endl;
    file << std::to_string(numMaxThreads) << std::endl;
    file << std::to_string(*columnLength) << std::endl;
    file << boolean                       << std::endl;

    for (int i=0; i < (numRows * numColumns); i++)
    {
        file << std::to_string(matrix[i]);
    }
    
    file.close();
}


// Retrieves a game from a saved status and reloads it into memory.
void loadGame()
{
    int difficulty = 0, numRows = 0, numColumns = 0, numMaxThreads = 0;
    int* columnLength = 0;
    bool automatic;
    FILE *file;
    
    // Open file to read.
    file = fopen("save.txt", "r");

    // Make sure it exists.
    if (file != NULL)
    {
        fscanf(file, "%i \n", &difficulty);
        fscanf(file, "%i \n", &numRows);
        fscanf(file, "%i \n", &numColumns);
        fscanf(file, "%i \n", &numMaxThreads);
        fscanf(file, "%i \n", columnLength);
    
        // Boolean extraction.
        int temp;
        fscanf(file, "%d", &temp);
        automatic = temp;
    
        // Write the matrix.
        int* matrix = (int*) malloc(numRows * numColumns * sizeof(int));
        for (int i=0; i < (numRows * numColumns); i++)
        {
            fscanf(file, "%i \n", &matrix[i]);
        }

        fclose(file);
        // Use all the variables just fetched from file to restart the game. 
            
        // EXECUTE GAME.
        playGame(difficulty, numRows, numColumns, numMaxThreads, columnLength,
                 automatic, matrix);
    }
    else
    {
        std::cout << "There is no file to load, try to start the application"
                  << "without charging the game!" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// -------------------------------- MAIN CODE ----------------------------------
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Variables needed as game settings.
    char mode;               // Game mode.
     int difficulty;         // Difficulty of the game.
     int numRows;            // Number of rows in the game.
     int numColumns;         // Number of columns in the game.
     int numMaxThreads;      // Number of max threads to be run.
    int* columnLength;       // The length of the bigger number in the column

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
        numColumns = std::stoi(argv[3]);
        numRows    = std::stoi(argv[4]);

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

        numMaxThreads = getThreadsMaxBlock();

        int recommendedCells = getMinBoard(difficulty); 
        
        if(numRows * numColumns > numMaxThreads)
        {
            std::cout << "The board is too big, please specify other board size."
                      << "The maximum number of cells is " << numMaxThreads
                      << std::endl;
            exit(0);
        } 
        else if(numRows * numColumns < recommendedCells)
        {
            std::cout << "The board is too small, please specify a bigger board"
                      << "size. " << std::endl << "The minimum for the level " 
                      << difficulty << " number of recommended minimum cells "
                      << "is " << recommendedCells << "." << std::endl;
            exit(0);
        }

        bool gameMode = (mode == 'a');

        // Matrix in which all the operations are going to take
        // place.
        int* matrix = (int*) calloc(numRows * numColumns, sizeof(int));
        
        std::cout << "Do you wish to load an earlier game? (y/n)?" << std::endl;
      
        bool load = true;

        while(load)
        {
            std::cin >> input;

            if (input.length() == 1)
            {
                switch(input[0])
                {
                    case 'y':
                        load = false;

                        loadGame();
                        break;

                    case 'n':
                        load = false;

                        std::cout << "Then, the game will continuw with the "
                                  << "specified settings!" << std::endl;

                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
 
                        // Initialization of the array which stores the length of the numbers in
                        // each column
                        columnLength = (int*) malloc(numColumns);
                        std::fill_n(columnLength, numColumns, 1);            

                        // EXECUTE GAME.
                        playGame(difficulty, numRows, numColumns, numMaxThreads, columnLength,
                             gameMode, matrix);
                        break;

                    default:
                        std::cout << "The response wasn't the expected!" << std::endl;
                        break;
                }
            }
            else
            {
                std::cout << "The response wasn't the expected!" << std::endl;
            }
        }

    }
}


// -----------------------------------------------------------------------------
// ----------------------------- CUDA METHODS-----------------------------------
// -----------------------------------------------------------------------------
cudaError_t cellsMerge(
    char movement,
    int  numRows, 
    int  numColumns, 
    int* matrix, 
    int* POINTS, 
    int* CELLS_OCCUPIED, 
    int* columnLength)
{
    // Variable to operate in the GPU
    int* dev_matrix = 0;
    int* dev_POINTS = 0;
    int* dev_CELLSO = 0;
    int* dev_colLen = 0;

    // Used to initialize the GRID and the BLOCK dimesions.
    int dimensionLength;

    if(movement == 'w' || movement == 's')
    {
        dimensionLength = numColumns;
    }
    else 
    {
        dimensionLength = numRows;
    }


    // GPU threads distribution
    dim3 dimGrid(dimensionLength / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, 1);

    // Selection of the GPU were code is to be executed
    cudaSetDevice(0);
    check_CUDA_Error("cudaSetDevice failed!\n");
    
    // Memory Allocation for GPU Variables
    cudaMalloc((void**) &dev_matrix, numRows * numColumns * sizeof(int));
    check_CUDA_Error("cudaMalloc failed at Matrix!\n");

    cudaMalloc((void**) &dev_POINTS, sizeof(int));
    check_CUDA_Error("cudaMalloc failed at POINTS!\n");

    cudaMalloc((void**) &dev_CELLSO, sizeof(int));
    check_CUDA_Error("cudaMalloc failed at CELLS_OCCUPIED!\n");

    cudaMalloc((void**) &dev_colLen, numColumns * sizeof(int));
    check_CUDA_Error("cudaMalloc failed at ColumnLength!\n");
    
    // Memory Transfer: CPU -> GPU
    cudaMemcpy(dev_matrix, matrix, numRows * numColumns * sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at Matrix (CPU -> GPU)!\n");
    
    cudaMemcpy(dev_POINTS, POINTS, sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at POINTS (CPU -> GPU)!\n");
    
    cudaMemcpy(dev_CELLSO, CELLS_OCCUPIED, sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at CELLS_OCCUPIED (CPU -> GPU)!\n");
    
    cudaMemcpy(dev_colLen, columnLength, numColumns * sizeof(int),
               cudaMemcpyHostToDevice);
    check_CUDA_Error("cudaMemCpy failed at columnLength (CPU -> GPU)!\n");
    
    /*
     * If the movement is UP or DOWN:
     *     The number of threads is the number of columns.
     * If the movement is LEFT or RIGHT 
     *     The number of threads is the number fo rows.
     * In any case, it is taken a certain TILE_WIDTH to optimize the execution
     * by using several blocks for the same operation.
     */
    switch(movement)
    {
        case 'w':
            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows,
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");

            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            computeMatrixUp<<<dimGrid, dimBlock>>>(numRows, numColumns, 
                                                   dev_matrix, dev_POINTS, 
                                                   dev_CELLSO, dev_colLen);
            check_CUDA_Error("Error merging cells!\n");
            
            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows,
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");
            break;

        case 's':
            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows, 
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");

            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            computeMatrixDown<<<dimGrid, dimBlock>>>(numRows, numColumns, 
                                                     dev_matrix, dev_POINTS, 
                                                     dev_CELLSO, dev_colLen);
            check_CUDA_Error("Error merging cells!\n");
            
            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows, 
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");
            break;
        
        case 'a':
            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows, 
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");

            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            computeMatrixLeft<<<dimGrid, dimBlock>>>(numRows, numColumns, dev_matrix, 
                                              dev_POINTS, dev_CELLSO,
                                              dev_colLen);
            check_CUDA_Error("Error merging cells!\n");
            
            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows, 
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");
            break;
        
        case 'd':
            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows, 
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");

            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            computeMatrixRight<<<dimGrid, dimBlock>>>(numRows,numColumns, dev_matrix,
                                               dev_POINTS, dev_CELLSO,
                                               dev_colLen);
            // Waits for kernel to finish
            cudaDeviceSynchronize();
            check_CUDA_Error("cudaDeviceSynchronize returned error!\n");

            check_CUDA_Error("Error merging cells!\n");
            
            fillSpace<<<dimGrid, dimBlock>>>(dev_matrix, movement, numRows, 
                                             numColumns);
            check_CUDA_Error("Error while gathering cells!\n");
            break;
    }
    
    // Waits for kernel to finish
    cudaDeviceSynchronize();
    check_CUDA_Error("cudaDeviceSynchronize returned error!\n");


    // Memory Transfer: GPU -> CPU
    cudaMemcpy(matrix, dev_matrix, numRows * numColumns * sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at Matrix (GPU -> CPU)!\n");

    cudaMemcpy(POINTS, dev_POINTS, sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at POINTS (GPU -> CPU)!\n");

    cudaMemcpy(CELLS_OCCUPIED, dev_CELLSO, sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at CELLS_OCCUPIED (GPU -> CPU)!\n");
    
    cudaMemcpy(columnLength, dev_colLen, numColumns * sizeof(int),
               cudaMemcpyDeviceToHost);
    check_CUDA_Error("cudaMemCpy failed at columnLength -> CPU)!\n");

    return cudaGetLastError();

}

