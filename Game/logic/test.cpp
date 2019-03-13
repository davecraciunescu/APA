#include <iostream>
#include <ctime>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

int     max;
int     win;
int     plus;
int     score;
int     apocalypse;
char    control;

int grid[4][4];
int bgrid[4][4];

// ----------------------------------------------------------------------------

int random_index(int x)
{
    int index;
    index = (rand() % x) + 0;

    return index;
}

// ----------------------------------------------------------------------------

void start_grid()
{
    int i, j;

    plus    = 0;
    score   = 0;
    max     = 0;

    for (i=0; i < 4; i++)
    {
        for (j=0; j < 4; j++)
        {
            grid[i][j] = 0;
        }
    }

    i = random_index(4);
    j = random_index(4);

    grid[i][j] = 2;

    i = random_index(4);
    j = random_index(4);

    grid[i][j] = 2;
}

// ----------------------------------------------------------------------------

int main()
{

    start_grid();
    
    system("clear");

    if (plus)
    {
        std::cout << "+" << plus << "!";
    }
    else
    {
        std::cout <<"    ";
    }

    std::cout << "\t\t\t\t\tSCORE::" << score << " \n\n\n\n";

    for (int i=0; i < 4; i++)
    {
        std::cout <<"\t\t    |";

        for (int j=0; j < 4; j++)
        {
            if (grid[i][j])
            {
                printf("%4d     |", grid[i][j]);
            }
            else
            {
                printf("%4c     |", ' ');
            }
        }

        std::cout << std::endl << std::endl;
    }
}

// ----------------------------------------------------------------------------

void update_grid()
{
    plus        = 0;
    apocalypse  = 1;

    // Iteration Limits.
    // Warning, these values are complicated to translate.
    int startPoint;
    int endPoint;

    switch (control)
    {
        for (int i=0; i < 4; i++)
        {
            for (int j=0; j < 3; j++)
            {
                
            }
        }

        case 'w':
            startPoint  = 0;
            endPoint    = 3;
                    

    }

}


void simple_update_grid()
{
    plus        = 0;
    apocalypse  = 1;

    switch (control)
    {
        case 'w':
            for (int i=0; i<4; i++)
                {
                    for (int j=0; j<3; j++)
                    {
                        if (grid[j][i] && (grid[j][i] == grid[j+1][i]))
                        {
                               apocalypse  = 0;
                            grid   [j][i] += grid [j+1][i];
                            grid [j+1][i]  = 0;
                                     plus += (((log2(grid[j][i])) - 1) * grid[j][i]);
                                    score += (((log2(grid[j][i])) - 1) * grid[j][i]);
                        }
                    }
                }
        break;

        case 's':
            
    }
}
