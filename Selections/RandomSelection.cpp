#include "RandomSelection.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>

__device__ void RandomSelection::selection(int* drawnChromosome, int* chromosome, int size, int populationSize) const
{
	float mainChromosomeFitness = 0; // calculateFitness(chromosome, size);
	float drawnChromosomeFitness = 0;// calculateFitness(drawnChromosome, size);
	if (mainChromosomeFitness > drawnChromosomeFitness)
	{
		for (int i = 0; i < size; ++i) {
			chromosome[i] = drawnChromosome[i];
		}
	}
}

__device__ void selection(int* drawnChromosome, int* chromosome, int size, float fitness1, float fitness2)
{
	if (fitness1 > fitness2)
	{
		for (int i = 0; i < size; ++i) {
			chromosome[i] = drawnChromosome[i];
		}
	}
}

__device__ int rouletteWheelSelection(float* fitness, int populationSize, curandState* state) {
	float totalFitness = 0;
	for (int i = 0; i < populationSize; ++i) {
		totalFitness += fitness[i];
	}

	float slice = curand_uniform(state) * totalFitness;
	float total = 0;
	for (int i = 0; i < populationSize; ++i) {
		total += fitness[i];
		if (total > slice) {
			return i;
		}
	}
	return populationSize - 1;
}