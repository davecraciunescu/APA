#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void toFile (
      int difficulty,
      int numRows,
      int numColumns,
      int numMaxThreads,
     int* columnLength,
     bool automatic,
     int* matrix
    )
{
    // File exportation variables.
    std::ofstream file;

    // Parse boolean value.
    std::string boolean = automatic ? "true":"false";

    file.open("test.txt");

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

// Reads values from file and loads them into the given values.
void fromFile ()
{
    int difficulty = 0, numRows = 0, numColumns = 0, numMaxThreads = 0, columnLength = 0;
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
        fscanf(file, "%i \n", &columnLength);
    
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
    }
}

int main()
{
    int var = 5;

    int* matrix;

    for (int i=0; i < 10; i++)
    {
        matrix[i] = i;
    }

    toFile(1, 2, 3, 4, &var, true, matrix);

    int* valores = fromFile();
}
