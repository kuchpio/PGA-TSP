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

__global__ void geneticAlgorithmKernel(int** population, float** distance_matrix, int size, curandState* globalState, int max_iterations) {
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
		selection(population[id], population[otherIdToSelection], size, calculateFitness(population[id], size, distance_matrix), calculateFitness(population[otherIdToSelection], size, distance_matrix));

		__syncthreads(); // Synchronize after selection for crossover

		int otherIdToCrossover = curand(&localState) % populationSize;
		int* child = crossover(population[id], population[otherIdToCrossover], size, &localState);
		for (int i = 0; i < size; ++i) {
			population[id][i] = child[i];
		}
		delete[] child;

		// Mutation
		mutate(&population[id * size], size, &localState);

		__syncthreads(); // Synchronize after mutation
	}
	// Update the global state to ensure randomness continuity
	globalState[id] = localState;
}

__global__ void tspGeneticAlgorithmKernel(int** population, float** distance_matrix, float* fitness, int size, int populationSize, curandState* globalState, int max_iterations) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int populationSize = blockDim.x * gridDim.x;
	// Local curand state
	curandState localState = globalState[id];

	// Initialize chromosome with a sequence from 0 to size - 1
	for (int i = 0; i < size; ++i) {
		population[id][i] = i;
	}

	shuffleChromosome(population[id], size, &localState);
	fitness[i] = calculateFitness(population[id], size, distance_matrix);

	for (int iteration = 0; iteration < max_iterations; ++iteration) {
		__syncthreads(); // Synchronize threads before selection

		// Selection - using roulette wheel to select an index
		int selectedIdx = rouletteWheelSelection(fitness, populationSize, &localState);

		// Crossover - Order Crossover (OX)
		int* parent1 = population[id];
		int* parent2 = population[selectedIdx];
		int* child = new int[size]; // Assuming dynamic memory allocation is allowed for illustrative purposes
		crossover(parent1, parent2, child, size, &localState);
		for (int i = 0; i < size; ++i) {
			population[id][i] = child[i]; // Copy the child to the current chromosome
		}
		delete[] child; // Clean up dynamically allocated memory

		__syncthreads(); // Synchronize threads before mutation

		// Mutation - Inversion Mutation
		inversionMutation(population[id], size, &localState);

		// Calculate fitness of the new chromosome
		fitness[i] = calculateFitness(population[id], size, distance_matrix);

		__syncthreads(); // Synchronize threads after mutation
	}

	// Update the global state to ensure randomness continuity
	globalState[id] = localState;
}