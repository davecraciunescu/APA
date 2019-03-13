#include <iostream>
#include <curses.h>
#include <chrono>
#include <thread>

/** KEY MOVEMENT DIFFERS FROM OPERATING SYSTEM TO OPERATING SYSTEM.
 * IN ORDER TO MAKE THIS WORK FOR WINDOWS MAKE THE FOLLOWING LINES.
 *
 * #include <conio.h>
 * ->remove #include <curses.h>
 *
 * #define KEY_UP    72
 * #define KEY_DOWN  80
 * #define KEY_LEFT  75
 * #define KEY_RIGHT 77
 *
*/

int main()
{    
    while (1)
    {
        if (getchar() == '\033')
        {
            getch();
            getch();
            switch(getch()) 
            {
                case 'A':
                    std::cout << std::endl << "UP" << std::endl;
                    break;
                case 'B':
                    std::cout << std::endl << "DOWN" << std::endl;
                    break;
                case 'C':
                    std::cout << std::endl << "RIGHT" << std::endl;
                    break;
                case 'D':
                    std::cout << std::endl << "LEFT" << std::endl;
                    break;
            }
        }
    }
}
