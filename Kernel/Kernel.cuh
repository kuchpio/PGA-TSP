#pragma once
#include <curand_kernel.h>
void geneticAlgorithmKernel(int* population, float** distance_matrix, int size, int populationSize, int max_iterations);

void tspGeneticAlgorithmKernel(int** population, float* distance_matrix, float* fitness, int size, int populationSize, curandState* globalState, int max_iterations);