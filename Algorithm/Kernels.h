#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Helper.h"
#include "../Selections/Basic.h"
#include "../Crossovers/Interval.h"
#include "../Selections/RouleteWheel.h"
#include "../Mutations/Basic.h"
#include "../Mutations/Interval.h"

namespace tsp {
	template <typename Instance>
	__global__ void geneticAlgorithmKernel(const Instance instance, int* fitness, int* population, curandState* globalState, int maxIterations) {
		__shared__ int totalFitness[256];
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int bid = threadIdx.x;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome = population + tid * instanceSize;

		initChromosome(chromosome, instanceSize, &localState);
		fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
		__syncthreads();

		int* result = new int[instanceSize];
		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			totalFitness[bid] = fitness[tid];
			__syncthreads();
			for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
				if (bid < stride) {
					totalFitness[bid] += totalFitness[bid + stride];
				}
				__syncthreads();
			}

			// Selection
			// Assuming a simplistic random selection for demonstration
			int otherIdToSelection = curand(&localState) % instanceSize;
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
			fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
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

		initChromosome(chromosome, instanceSize, &localState);
		fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
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
			fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
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

		initChromosome(chromosome, instanceSize, &localState);
		fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
		__syncthreads();

		for (int iteration = 0; iteration < maxIterations; ++iteration) {
			sharedFitness[bid] = fitness[tid];
			sharedIndexes[bid] = bid;
			totalFitness[bid] = fitness[tid];
			__syncthreads();
			for (int stride = blockDim.x >> 2; stride > 0; stride >>= 1) {
				if (bid < stride) {
					if (sharedFitness[bid] < sharedFitness[bid + stride]) {
						sharedFitness[bid] = sharedFitness[bid + stride];
						sharedIndexes[bid] = sharedIndexes[bid + stride];
					}
					totalFitness[bid] += totalFitness[bid + stride];
				}
				__syncthreads();
			}
			__syncthreads();
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
			fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
			__syncthreads();
		}

		__syncthreads();
		globalState[tid] = localState;
		delete[] result;
	}
}

#endif