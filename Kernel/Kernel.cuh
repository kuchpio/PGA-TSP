#pragma once
#include <curand_kernel.h>
void shuffleChromosome(int* chromosome, int size, curandState* state);

void geneticAlgorithmKernel(int** population, float** distance_matrix, int size, curandState* globalState, int max_iterations);

void tspGeneticAlgorithmKernel(int** population, float** distance_matrix, float* fitness, int size, int populationSize, curandState* globalState, int max_iterations);