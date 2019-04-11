### Práctica 1. Ampliación de Programación Avanzada.

##### Pablo Acereda García, con la colaboración de David Emanuel Craciunescu.

***

### Descripción del trabajo a realizar

La práctica propuesta en la actividad 1 consiste en la realización del "clásico" juego de **2048** empleando las diferentes herramientas que CUDA proporciona. Se podría decir que la idea fundamental de la práctica se basa en el buen diseño de los sistemas de índices de los diferentes núcleos CUDA empleados para que la labor de ninguno de estos interfiera con cualquier otro.

### Explicación del trabajo realizado.

Dentro de la práctica, los aspectos más importantes, y considerados de mención son los siguientes:

* Desarrollo de movimientos en base a los cores de CUDA.
* Diseño de interfaz gráfica empleando ASCII art para que el usuario se pueda sumergir en una revolucionaria experiencia visual.
* Cuidadosa planificación de las diferentes acciones de la aplicación, debido a la complejidad del propio código, dada la naturaleza de la programación en CUDA.
* Sistema de guardado de partida. El jugador no deberá preocuparse por las situaciones externas al juego, dado que este será capaz de pasar la ejecución del programa y guardar su partida en cualquier momento.
* Sistema de vidas, para fomentar la competitividad entre jugadores.

### Código CUDA representativo del ejercicio.

En esta sección se puede encontrar un ejemplo representativo de las diferentes acciones que el programa puede realizar.

```C++
__global__ void computeMatrixRight ( int numRows,
                                     int numColumns,
                                    int* matrix,
                                    int* POINTS,
                                    int* CELLS_OCCUPIED,
                                    int* columnLength)
{
    // Matrix dimensions.
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Location in matrix.
    int row = bx * TILE_WIDTH + tx; // → Classical CUDA Matrix Indexation.

    if (row < numRows)
    {
        for (int i = numColumns -1; i > 0; i--)
        {
            if (matrix[row * numRows + i] > 0 && matrix[row * numRows + i] == matrix[row * numRows + (i - 1)])
            {
                matrix[row * numRows + i] *= 2;
                matrix[row * numRows + (i - 1)] = 0;

                atomicAdd(POINTS, matrix[row * numRows + i]);
                atomicAdd(CELLS_OCCUPIED, -1);
            }
        }
    }
}
```
