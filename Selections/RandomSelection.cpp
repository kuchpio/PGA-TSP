#include "RandomSelection.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>

__device__ void RandomSelection::selection(int* drawnChromosome, int* chromosome, int size, int populationSize) const
{
	float mainChromosomeFitness = calculateFitness(chromosome, size);
	float drawnChromosomeFitness = calculateFitness(drawnChromosome, size);
	if (mainChromosomeFitness > drawnChromosomeFitness)
	{
		for (int i = 0; i < size; ++i) {
			chromosome[i] = drawnChromosome[i];
		}
	}
}