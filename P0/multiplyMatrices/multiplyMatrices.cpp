#include "pch.h"
#include <iostream>

int main()
{
	// Estado de ejecucion
	int state = 0;
	// Dimensiones de las matrices
	int col1 = 3, col2 = 2, row1 = 2, row2 = 3;
	// Se inicializa el numero de columnas
	int** matrix1 = (int**) malloc(row1 * sizeof(int*));
	int** matrix2 = (int**) malloc(row2 * sizeof(int*));
	// Se inicializanas filas de cada matriz
	for (int i = 0; i < row1; i++)
	{
		matrix1[i] = (int*) malloc(col1 * sizeof(int));
	}

	for (int i = 0; i < row2; i++)
	{
		matrix2[i] = (int*) malloc(col2 * sizeof(int));
	}
	// Se inicializan las matrices
	matrix1[0][0] = 1;
	matrix1[0][1] = 2;
	matrix1[0][2] = 3;
	matrix1[1][0] = 4;
	matrix1[1][1] = 5;
	matrix1[1][2] = 6;

	matrix2[0][0] = 5;
	matrix2[0][1] = -1;
	matrix2[1][0] = 1;
	matrix2[1][1] = 0;
	matrix2[2][0] = -2;
	matrix2[2][1] = 3;
	/*
	 * Se pueden multiplicar matrices en caso de que el numero de 
	 * columnas de la primera matriz coincida con el numero de filas
	 * de la seguna matriz
	 *
	 *             | 5  -1 |
	 * | 1 2 3 | · | 1   0 | = |  1  8 |
	 * | 4 5 6 |   |-2   3 |   | 13 14 |
	 *
	 */
	if (col1 == row2)
	{
		// Se reserva espacio para la matriz que almacena el resultado
		int** resultMatrix = (int**) calloc(row1, sizeof(int*));

		for (int i = 0; i < row1; i++)
		{
			resultMatrix[i] = (int*) calloc(col2, sizeof(int*));
		}

		

		// Multiplicacion de matrices
		for (int i = 0; i < row1; i++)
		{
			for(int j = 0; j < col2; j++)
			{
				for (int k = 0; k < col1; k++)
				{
					resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
				}
			}
		}

		// Se imprimen los datos de la matriz resultante por pantalla
		for (int i = 0; i < row1; i++)
		{
			for (int j = 0; j < col2; j++)
			{
				std::cout << resultMatrix[i][j];

				if (j != col2 - 1) std::cout << "-";
			} 

			std::cout << std::endl;
		}

		free(resultMatrix);

		state = 0;
	}
	else 
	{
		std::cout << "Las matrices no se puede multiplicar: el numero de columnas de la matriz 1 ("
				  << col1 << ") es diferente al numero de filas de la seguna matriz ("
				  << row2 << ")" << std::endl;

		state = -1;
	}

	free(matrix1);
	free(matrix2);
	
	return state;
}
