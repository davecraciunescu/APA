#include "pch.h"
#include <iostream>

int main()
{
	// Estado de ejecucion del script
	int state;
	// Tamaño que van a ocupar los vectores
	int size1 = 5, size2 = 5;
	// Se generan los vectores reservando espacio para los mismos
	int* vect1 = (int*)malloc(size1 * sizeof(int));
	int* vect2 = (int*)malloc(size2 * sizeof(int));
	// Valores inicializados
	vect1[0] = 1;
	vect1[1] = 2;
	vect1[2] = 3;
	vect1[3] = 4;
	vect1[4] = 5;

	vect2[0] = 8;
	vect2[1] = 9;
	vect2[2] = 11;
	vect2[3] = 13;
	vect2[4] = 15;

	if (size1 == size2)
	{
		// Vector que almacena el resultado
		int* vectResult = (int*)malloc(size1 * sizeof(int));
		// Sumatorio de los correspondientes datos
		for (int i = 0; i < size1; i++)
		{
			// En caso de que los datos deban ser utilizados con posterioridad, son almacenados
			vectResult[i] = vect1[i] + vect2[i];

			std::cout << vectResult[i];

			if (i != size1 - 1) std::cout << "-";
		}

		std::cout << std::endl;

		free(vectResult);

		state = 0;

	}
	else
	{
		std::cout << "El numero de casillas entre los vectores no es el mismo, ERROR!" << std::endl;

		state = -1;
	}

	free(vect1);
	free(vect2);

	return state;
}