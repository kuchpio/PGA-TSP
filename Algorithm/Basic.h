#ifndef __ALGORITHM_BASIC_H__
#define __ALGORITHM_BASIC_H__
#define MAX_ITER 50000
#define SEED 100

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "Helper.h"
#include "../Selections/Basic.h"
#include "../Crossovers/Basic.h"
#include "../Mutations/Basic.h"

namespace tsp {
	template <typename Instance>
	__global__
		void solveTSPBasicKernel(const Instance instance, int* fitness, int* population, curandState* globalState, int maxIterations) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int instanceSize = size(instance);
		curandState localState = globalState[tid];
		int* chromosome = population + tid * instanceSize;
		int* result = new int[instanceSize];

		initChromosome(chromosome, instanceSize, &localState);
		fitness[tid] = hamiltonianCycleWeight(instance, chromosome);

		for (int i = 0; i < maxIterations; i++) {
			// Selection
			__syncthreads();
			int selectionCandidateTid = curand(&localState) % blockDim.x + blockIdx.x * blockDim.x;
			select(chromosome, population + selectionCandidateTid * instanceSize, result,
				instanceSize, fitness[tid], fitness[selectionCandidateTid]);
			__syncthreads();
			for (int i = 0; i < instanceSize; i++)
				chromosome[i] = result[i];

			// Crossover
			__syncthreads();
			int crossoverParentTid = curand(&localState) % blockDim.x + blockIdx.x * blockDim.x;
			crossover(chromosome, population + crossoverParentTid * instanceSize, result, instanceSize, fitness[tid], fitness[crossoverParentTid]);
			__syncthreads();
			for (int i = 0; i < instanceSize; i++)
				chromosome[i] = result[i];

			// Mutation
			mutate(chromosome, instanceSize, &localState);

			// Fitness
			fitness[tid] = hamiltonianCycleWeight(instance, chromosome);
		}

		delete[] result;
	}

	template <typename Instance>
	int solveTSP(const Instance instance) {
		const int blockSize = 256, gridSize = 4;
		int* d_fitness, * d_population;
		curandState* d_globalState;

		if (cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState)) != cudaSuccess)
			return -1;

		setupCurand << <gridSize, blockSize >> > (d_globalState, SEED);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		solveTSPBasicKernel << <gridSize, blockSize >> > (instance, d_fitness, d_population, d_globalState, MAX_ITER);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		auto fitnessPtr = thrust::device_pointer_cast(d_fitness);
		int opt = MAX_DISTANCE_CAN - *thrust::max_element(fitnessPtr, fitnessPtr + blockSize * gridSize);

		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}

	template <typename Instance>
	int solveTSP2(const Instance instance) {
		const int blockSize = 256, gridSize = 4;
		int* d_fitness, * d_population;
		curandState* d_globalState;

		if (cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState)) != cudaSuccess)
			return -1;

		setupCurand << <gridSize, blockSize >> > (d_globalState, SEED);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		geneticAlgorithmKernel << <gridSize, blockSize >> > (instance, d_fitness, d_population, d_globalState, MAX_ITER);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		auto fitnessPtr = thrust::device_pointer_cast(d_fitness);
		int opt = MAX_DISTANCE_CAN - *thrust::max_element(fitnessPtr, fitnessPtr + blockSize * gridSize);
		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}

	template <typename Instance>
	int solveTSP3(const Instance instance) {
		const int blockSize = 256, gridSize = 4;
		int* d_fitness, * d_population;
		curandState* d_globalState;

		if (cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState)) != cudaSuccess)
			return -1;

		setupCurand << <gridSize, blockSize >> > (d_globalState, SEED);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		tspGeneticAlgorithmKernel << <gridSize, blockSize >> > (instance, d_fitness, d_population, d_globalState, MAX_ITER);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		auto fitnessPtr = thrust::device_pointer_cast(d_fitness);
		int opt = MAX_DISTANCE_CAN - *thrust::max_element(fitnessPtr, fitnessPtr + blockSize * gridSize);

		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}

	template <typename Instance>
	int solveTSP4(const Instance instance) {
		const int blockSize = 256, gridSize = 4;
		int* d_fitness, * d_population;
		curandState* d_globalState;

		if (cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState)) != cudaSuccess)
			return -1;

		setupCurand << <gridSize, blockSize >> > (d_globalState, SEED);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		tspElitistGeneticAlgorithmKernel << <gridSize, blockSize >> > (instance, d_fitness, d_population, d_globalState, MAX_ITER);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		auto fitnessPtr = thrust::device_pointer_cast(d_fitness);
		int opt = MAX_DISTANCE_CAN - *thrust::max_element(fitnessPtr, fitnessPtr + blockSize * gridSize);
		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}

	template <typename Instance>
	int solveTSP5(const Instance instance) {
		const int blockSize = 256, gridSize = 4;
		int* d_fitness, * d_population;
		curandState* d_globalState;

		if (cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int)) != cudaSuccess)
			return -1;

		if (cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState)) != cudaSuccess)
			return -1;

		setupCurand << <gridSize, blockSize >> > (d_globalState, SEED);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		tspRandomStepAlgorithmKernel << <gridSize, blockSize / 2 >> > (instance, d_fitness, d_population, d_globalState, MAX_ITER);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return -1;

		auto fitnessPtr = thrust::device_pointer_cast(d_fitness);
		int opt = MAX_DISTANCE_CAN - *thrust::max_element(fitnessPtr, fitnessPtr + blockSize * gridSize);
		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}
}

#endif
