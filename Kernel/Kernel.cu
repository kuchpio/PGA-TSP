#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "./Kernel.cuh"
#include "../Selections/RandomSelection.h"
#include "../Crossovers/IntervalCrossover.h"
#include "../Mutations/SwapMutation.h"

// Fisher-Yates shuffle algorithm
__device__ void shuffleChromosome(int* chromosome, int size, curandState* state) {
	for (int i = size - 1; i > 0; i--) {
		int j = curand(state) % (i + 1);

		// Swap chromosome[i] with chromosome[j]
		int temp = chromosome[i];
		chromosome[i] = chromosome[j];
		chromosome[j] = temp;
	}
}

__global__ void geneticAlgorithmKernel(int** population, float* distance_matrix, int size, curandState* globalState, int max_iterations) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int populationSize = blockDim.x * gridDim.x;
	// Local curand state
	curandState localState = globalState[id];

	// Initialize chromosome with a sequence from 0 to size - 1
	for (int i = 0; i < size; ++i) {
		population[id][i] = i;
	}

	shuffleChromosome(population[id], size, &localState);

	// Run the genetic algorithm for a fixed number of iterations
	for (int iteration = 0; iteration < max_iterations; ++iteration) {
		// Selection
		// Assuming a simplistic random selection for demonstration
		int otherIdToSelection = curand(&localState) % populationSize;
		RandomSelection::selection(population[id], population[otherIdToSelection], size, populationSize);

		__syncthreads(); // Synchronize after selection for crossover

		int otherIdToCrossover = curand(&localState) % populationSize;
		int* child = IntervalCrossover::crossover(population[id], population[otherIdToCrossover], size, &localState);
		for (int i = 0; i < size; ++i) {
			population[id][i] = child[i];
		}
		delete[] child;

		// Mutation
		SwapMutation::mutate(&population[id * size], size, &localState);

		__syncthreads(); // Synchronize after mutation
	}
}