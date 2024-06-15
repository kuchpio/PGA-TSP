#ifndef __ALGORITHM_COARSE_GRAINED_H__
#define __ALGORITHM_COARSE_GRAINED_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "Kernels.h"
#include "Helper.h"
#include "BlockSwap.h"

namespace tsp {

	template <typename Instance>
	int solveTSPCoarseGrained(const Instance instance, struct IslandGeneticAlgorithmOptions options, int *globalBestCycle, int seed)
	{
		cudaError status;
		const int blockSize = 256, gridSize = options.islandCount;
		int* d_fitness, * d_population, * d_bestFitness;
		int* h_fitness = new int[blockSize * gridSize];
		int* h_bestFitness = new int[gridSize];
		int opt;
		curandState* d_globalState;
		unsigned int stalledMigrationsCount = 0, stalledBestCycleWeight = (unsigned int)-1;

		if ((status = cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMemset(d_fitness, -1, blockSize * gridSize * sizeof(int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_bestFitness, gridSize * sizeof(int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		setupCurand << <gridSize, blockSize >> > (d_globalState, seed);

		if ((status = cudaGetLastError()) != cudaSuccess) {
			std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		for (unsigned int i = 0; i < options.migrationCount; ++i)
		{
			tspRandomStepAlgorithmKernel << <gridSize, blockSize / 2 >> > (instance, d_fitness, d_population, d_bestFitness, d_globalState, 
				options.isolatedIterationCount, options.stalledIsolatedIterationsLimit, options.crossoverProbability, options.mutationProbability, options.elitism);

			if ((status = cudaGetLastError()) != cudaSuccess) {
				std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaMemcpy(h_fitness, d_fitness, blockSize * gridSize * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaMemcpy(h_bestFitness, d_bestFitness, gridSize * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			updateStalledMigrationsCount(stalledMigrationsCount, stalledBestCycleWeight, (unsigned int*)h_fitness, (unsigned int*)h_bestFitness, gridSize, blockSize);
			if (stalledMigrationsCount >= options.stalledMigrationsLimit) break;

			ChangeBestChromosomes << <gridSize, blockSize >> > (instance, d_population, d_bestFitness, d_globalState);

			if ((status = cudaGetLastError()) != cudaSuccess) {
				std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}
		}

		tspRandomStepAlgorithmKernel << <gridSize, blockSize / 2 >> > (instance, d_fitness, d_population, d_bestFitness, d_globalState, 
			options.isolatedIterationCount, options.stalledIsolatedIterationsLimit, options.crossoverProbability, options.mutationProbability, options.elitism);

		if ((status = cudaGetLastError()) != cudaSuccess) {
			std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		// Find best cycle
		{
			auto bestfitnessPtr = thrust::device_pointer_cast(d_bestFitness);
			int bestCycleIndex = *thrust::max_element(bestfitnessPtr, bestfitnessPtr + gridSize);
			if ((status = cudaMemcpy(globalBestCycle, d_population + bestCycleIndex * size(instance), sizeof(int) * size(instance), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}
			int bestCycleFitness = 0;
			if ((status = cudaMemcpy(&bestCycleFitness, d_fitness + bestCycleIndex, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}
			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}
			opt = MAX_DISTANCE_CAN - bestCycleFitness;
		}

		if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}
FREE:
		delete[] h_fitness;
		delete[] h_bestFitness;
		cudaFree(d_fitness);
		cudaFree(d_bestFitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}

}

#endif
