#include <iostream>
#include <fstream>
#include <string>

int main()
{
    std::string text;
    std::ofstream file;

    file.open("test.txt");

    std::cout << "Type some text" << std::endl;
    getline(std::cin, text);

    file << text;

    return 0;
}
