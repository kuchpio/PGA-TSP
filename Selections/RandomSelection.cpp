#include "RandomSelection.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>

__device__ __host__ void RandomSelection::selection(int** population, int* chromosome, int size, int populationSize, curandState* state) const
{
	int index = curand(state) % populationSize;
	int* drawnChromosome = population[index];
	float mainChromosomeFitness = calculateFitness(chromosome, size);
	float drawnChromosomeFitness = calculateFitness(drawnChromosome, size);
	if (mainChromosomeFitness > drawnChromosomeFitness)
	{
		for (int i = 0; i < size; ++i) {
			chromosome[i] = drawnChromosome[i];
		}
	}
}