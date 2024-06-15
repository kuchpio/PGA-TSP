#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Helper.h"
#include "CycleHelper.h"
#include "../Selections/Basic.h"
#include "../Crossovers/Basic.h"
#include "../Crossovers/Interval.h"
#include "../Crossovers/PMX.h"
#include "../Selections/RouleteWheel.h"
#include "../Mutations/Basic.h"
#include "../Mutations/Interval.h"
#include "./VectorReduction.h"

namespace tsp {
	template <typename Instance>
	__global__ void geneticAlgorithmKernel(const Instance instance, int* fitness, int* population, curandState* globalState, int maxIterations) {
		__shared__ int totalFitness[256];
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int bid = threadIdx.x;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome = population + tid * instanceSize;

		initializeCycle(chromosome, instanceSize, &localState);
		fitness[tid] = calculateCycleWeight(chromosome, instance);
		__syncthreads();

		int* result = new int[instanceSize];
		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			// Vector Reduction
			SumVector(totalFitness, fitness);

			// Selection
			// Assuming a simplistic random selection for demonstration
			int selectedIdx = rouletteWheelSelection(fitness, instanceSize, &localState, totalFitness[0]);

			//// Crossover - Order Crossover (OX)
			__syncthreads();
			// Crossover
			//int* child = intervalCrossover(chromosome, population + selectedIdx * instanceSize, instanceSize, &localState);
			crossover(chromosome, population + selectedIdx * instanceSize, result, instanceSize, fitness[tid], fitness[selectedIdx]);

			__syncthreads();
			for (int i = 0; i < instanceSize; ++i) {
				chromosome[i] = result[i];
			}
			__syncthreads();

			// Mutation
			if (curand_uniform(&localState) > 0.9) { // 10% chance of mutation
				mutate(chromosome, instanceSize, &localState);
			}
			fitness[tid] = calculateCycleWeight(chromosome, instance);
		}

		globalState[tid] = localState;
		delete[] result;
	}

	template <typename Instance>
	__global__ void tspGeneticAlgorithmKernel(const Instance instance, int* fitness, int* population, curandState* globalState, int maxIterations) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome = population + tid * instanceSize;
		int* result = new int[instanceSize];

		initializeCycle(chromosome, instanceSize, &localState);
		fitness[tid] = calculateCycleWeight(chromosome, instance);
		__syncthreads();

		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			// Selection
			int selectedIdx = curand(&localState) % instanceSize;
			select(chromosome, population + selectedIdx * instanceSize, result,
				instanceSize, fitness[tid], fitness[selectedIdx]);
			__syncthreads();
			// Crossover
			crossover(chromosome, population + selectedIdx * instanceSize, result, instanceSize, fitness[tid], fitness[selectedIdx]);
			__syncthreads();
			for (int i = 0; i < instanceSize; ++i) {
				chromosome[i] = result[i];
			}
			__syncthreads();

			// Mutation
			if (curand_uniform(&localState) > 0.9) { // 10% chance of mutation
				mutate(chromosome, instanceSize, &localState);
			}
			fitness[tid] = calculateCycleWeight(chromosome, instance);
			__syncthreads(); // Synchronize after mutation
		}

		globalState[tid] = localState;
		delete[] result;
	}

	template <typename Instance>
	__global__ void tspElitistGeneticAlgorithmKernel(const Instance instance, int* fitness, int* population, curandState* globalState, int maxIterations) {
		__shared__ int sharedFitness[256];
		__shared__ int sharedIndexes[256];
		__shared__ int totalFitness[256];
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int bid = threadIdx.x;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome = population + tid * instanceSize;
		int* result = new int[instanceSize];

		initializeCycle(chromosome, instanceSize, &localState);
		fitness[tid] = calculateCycleWeight(chromosome, instance);
		__syncthreads();

		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			// Vector Reduction
			SumAndGetMaxVector(totalFitness, fitness, sharedFitness, sharedIndexes);
			// Selection
			int selectedIdx = rouletteWheelSelection(fitness, blockDim.x * gridDim.x, &localState, totalFitness[0]);
			__syncthreads();
			// Crossover
			if (bid != sharedIndexes[0])
			{
				crossover(chromosome, population + selectedIdx * instanceSize, result, instanceSize, fitness[tid], fitness[selectedIdx]);
			}

			__syncthreads();

			if (bid != sharedIndexes[0])
			{
				for (int i = 0; i < instanceSize; ++i) {
					chromosome[i] = result[i];
				}
			}
			__syncthreads();

			// Mutation
			if (curand_uniform(&localState) > 0.9 && bid != sharedIndexes[0]) { // 10% chance of mutation
				mutate(chromosome, instanceSize, &localState);
			}
			fitness[tid] = calculateCycleWeight(chromosome, instance);
			__syncthreads();
		}

		__syncthreads();
		globalState[tid] = localState;
		delete[] result;
	}

	template <typename Instance>
	__global__ void tspPMXAlgorithmKernel(const Instance instance, int* fitness, int* population, curandState* globalState, int maxIterations) {
		__shared__ int totalFitness[128];
		int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
		int bid = threadIdx.x;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome[2] = { population + tid * instanceSize, population + (tid + 1) * instanceSize };
		int* result[2];
		result[0] = new int[instanceSize];
		result[1] = new int[instanceSize];
		bool flags[] = { false, false };
		for (int i = 0; i < 2; ++i) {
			initializeCycle(chromosome[i], instanceSize, &localState);
			fitness[tid + i] = calculateCycleWeight(chromosome[i], instance);
		}
		__syncthreads();

		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			// Vector Reduction
			SumVector(totalFitness, fitness);

			// Selection
			for (int i = 0; i < 2; ++i) {
				flags[i] = false;
				int selectedIdx = rouletteWheelSelection(fitness, instanceSize, &localState, totalFitness[0]);
				if (fitness[selectedIdx] > fitness[tid + i])
				{
					flags[i] = true;
					int* betterChromosome = population + selectedIdx * instanceSize;
					for (int j = 0; j < instanceSize; ++j) {
						result[i][j] = betterChromosome[j];
					}
				}
			}
			__syncthreads();
			for (int i = 0; i < instanceSize; ++i) {
				if (flags[0])
					chromosome[0][i] = result[0][i];
				if (flags[1])
					chromosome[1][i] = result[1][i];
			}
			__syncthreads();
			// Crossover
			PMX(chromosome[0], chromosome[1], instanceSize, &localState);
			__syncthreads();
			// Mutation
			for (int i = 0; i < 2; ++i) {
				if (curand_uniform(&localState) > 0.7) { // 10% chance of mutation
					mutate(chromosome[i], instanceSize, &localState);
				}
				fitness[tid + i] = calculateCycleWeight(chromosome[i], instance);
			}
			__syncthreads(); // Synchronize after mutation
		}

		globalState[tid] = localState;
		globalState[tid + 1] = localState;
		delete[] result[0];
		delete[] result[1];
	}

	template <typename Instance>
	__global__ void tspRandomStepAlgorithmKernel(const Instance instance, int* fitness, int* population, int* bestFitness, curandState* globalState, 
		int maxIterations, int maxIterationStop, float crossoverProbability, float mutationProbability, bool elitism) 
	{
		__shared__ int totalFitness[128];
		__shared__ int sharedFitness[128];
		__shared__ int sharedFitnessIndex[128];
		__shared__ int commonRandomStep;
		__shared__ int counter;
		int prevFitness = 0;
		int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
		int bid = threadIdx.x;
		bool flag = false;
		if (bid == 0)
		{
			counter = 0;
		}
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome[2] = { population + tid * instanceSize, population + (tid + 1) * instanceSize };
		int* result[2];
		result[0] = new int[instanceSize];
		result[1] = new int[instanceSize];
		bool flags[] = { false, false };

		if (fitness[tid] == -1) {
			for (int i = 0; i < 2; ++i) {
				initializeCycle(chromosome[i], instanceSize, &localState);
				fitness[tid + i] = calculateCycleWeight(chromosome[i], instance);
			}
		}

		__syncthreads();

		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			// Vector Reduction
			SumVectorForChromosomes(totalFitness, fitness, sharedFitness, sharedFitnessIndex);
			if (counter == maxIterationStop)
			{
				break;
			}
			else if (bid == 0)
			{
				if (prevFitness == sharedFitness[0])
					++counter;
				else
					counter = 0;
				prevFitness = sharedFitness[0];
			}
			__syncthreads();
			// Selection
			for (int i = 0; i < 2; ++i) {
				flags[i] = false;
				int selectedIdx = rouletteWheelSelectionForChromosomes(fitness, instanceSize, &localState, totalFitness[0]);
				if (!(elitism && tid + i == sharedFitnessIndex[0]) && fitness[selectedIdx] > fitness[tid + i])
				{
					flags[i] = true;
					int* betterChromosome = population + selectedIdx * instanceSize;
					for (int j = 0; j < instanceSize; ++j) {
						result[i][j] = betterChromosome[j];
					}
				}
			}
			__syncthreads();
			for (int i = 0; i < instanceSize; ++i) {
				if (flags[0])
					chromosome[0][i] = result[0][i];
				if (flags[1])
					chromosome[1][i] = result[1][i];
			}
			__syncthreads();
			// Crossover
			if (!(elitism && (tid == sharedFitnessIndex[0] || tid + 1 == sharedFitnessIndex[0])) && curand_uniform(&localState) < crossoverProbability) {
				PMX(chromosome[0], chromosome[1], instanceSize, &localState);
			}
			__syncthreads();
			// Mutation
			for (int i = 0; i < 2; ++i) {
				if (!(elitism && tid + i == sharedFitnessIndex[0]) && curand_uniform(&localState) < mutationProbability) {
					mutate(chromosome[i], instanceSize, &localState);
				}
			}
			__syncthreads();
			if (bid == 0) {
				commonRandomStep = curand(&localState) % 128;
			}
			__syncthreads();
			int newIndx = blockDim.x * blockIdx.x * 2 + ((bid + commonRandomStep) * 2 + 1) % (blockDim.x * 2);
			if (!flag)
			{
				flag = !flag;
				chromosome[0] = population + newIndx * instanceSize;
				chromosome[1] = population + tid * instanceSize;
			}
			else
			{
				flag = !flag;
				chromosome[1] = population + newIndx * instanceSize;
				chromosome[0] = population + tid * instanceSize;
			}

			fitness[tid] = calculateCycleWeight(chromosome[0], instance);
			fitness[tid + 1] = calculateCycleWeight(chromosome[1], instance);
			__syncthreads();
		}

		SumVectorForChromosomes(totalFitness, fitness, sharedFitness, sharedFitnessIndex);

		if (bid == 0)
			bestFitness[blockIdx.x] = sharedFitnessIndex[0];

		globalState[tid] = localState;
		globalState[tid + 1] = localState;
		delete[] result[0];
		delete[] result[1];
	}
}

#endif