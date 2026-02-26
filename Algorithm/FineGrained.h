#ifndef __ALGORITHM_FINE_GRAINED_H__
#define __ALGORITHM_FINE_GRAINED_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iomanip>
#include <mpi.h>
#include <sstream>
#include <limits>

#include "Helper.h"
#include "WarpCycleHelper.h"
#include "../Selections/WarpRouletteWheel.h"
#include "../Crossovers/WarpPMX.h"
#include "../Mutations/WarpInterval.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace tsp {

	template <typename Instance, typename gene>
	__global__ void initializationKernel(const Instance instance, curandState* globalState, gene* population, unsigned int islandPopulationSize, 
		unsigned int* cycleWeight, unsigned int* islandBest, unsigned int* islandWorst) 
	{
		extern __shared__ unsigned int s_buffer[];
		unsigned int* s_reductionBuffer = s_buffer;
		unsigned int *s_cycleWeight = s_buffer + 4 * WARP_SIZE;

		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);			// Warp thread id
		unsigned int n = size(instance);
		unsigned int nWarpSizeAligned = (n & ~(WARP_SIZE - 1)) + WARP_SIZE;

		for (unsigned int chromosomeIndex = threadIdx.x / WARP_SIZE; chromosomeIndex < islandPopulationSize; chromosomeIndex += (blockDim.x / WARP_SIZE)) {
			gene* chromosome = population + (blockIdx.x * 2 * islandPopulationSize + chromosomeIndex) * nWarpSizeAligned;

			warpInitializeCycle(chromosome, n, globalState + tid);

			unsigned int thisCycleWeight = warpCalculateCycleWeight(chromosome, instance);
			if (lid == 0) s_cycleWeight[chromosomeIndex] = thisCycleWeight;
		}

		__syncthreads();

		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x)
			cycleWeight[blockIdx.x * islandPopulationSize + chromosomeIndex] = s_cycleWeight[chromosomeIndex];

		findMinMax(s_cycleWeight, islandPopulationSize, s_reductionBuffer, threadIdx.x, threadIdx.x, islandBest + blockIdx.x, islandWorst + blockIdx.x);
	}

	template <typename gene>
	__global__ void emigrationKernel(gene* population, unsigned int islandPopulationSize, unsigned int nWarpSizeAligned,
		unsigned int* islandBest, bool *sourceInSecondBuffer, gene* emigrationBuffer)
	{
		__shared__ gene *s_srcDst[2];

		unsigned int thisIsland = blockIdx.x;

		if (threadIdx.x == 0) {
			unsigned int thisIslandBestIndex = islandBest[thisIsland];
			bool thisSourceInSecondBuffer = sourceInSecondBuffer[thisIsland];

			s_srcDst[0] = population + nWarpSizeAligned *
				(2 * thisIsland * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + thisIslandBestIndex);
			s_srcDst[1] = emigrationBuffer + nWarpSizeAligned * thisIsland;
		}

		__syncthreads();

		gene* srcChromosome = s_srcDst[0];
		gene* dstChromosome = s_srcDst[1];

		for (unsigned int i = threadIdx.x; i < nWarpSizeAligned; i += blockDim.x)
			dstChromosome[i] = srcChromosome[i];
	}

	template <typename Instance, typename gene>
	__global__ void immigrationKernel(const Instance instance, gene* population, unsigned int islandPopulationSize, unsigned int nWarpSizeAligned,
		unsigned int *cycleWeight, unsigned int* islandWorst, bool *sourceInSecondBuffer, gene* immigrationBuffer)
	{
		__shared__ gene *s_srcDst[2];

		unsigned int thisIsland = blockIdx.x;
		unsigned int thisIslandWorstIndex = 0;

		if (threadIdx.x == 0) {
			thisIslandWorstIndex = islandWorst[thisIsland];
			bool thisSourceInSecondBuffer = sourceInSecondBuffer[thisIsland];

			s_srcDst[0] = immigrationBuffer + nWarpSizeAligned * thisIsland;
			s_srcDst[1] = population + nWarpSizeAligned *
				(2 * thisIsland * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + thisIslandWorstIndex);
		}

		__syncthreads();

		gene* srcChromosome = s_srcDst[0];
		gene* dstChromosome = s_srcDst[1];

		// Update cycle weight on new chromosome (first warp)
		if (threadIdx.x < warpSize) {
			auto thisCycleWeight = warpCalculateCycleWeight(srcChromosome, instance);

			if (threadIdx.x == 0) {
				unsigned int thisIslandWorstIndexGlobal = thisIsland * islandPopulationSize + thisIslandWorstIndex;
				cycleWeight[thisIslandWorstIndexGlobal] = thisCycleWeight;
			}
		} else { // Replace worst chromosome (other warps)
			for (unsigned int i = threadIdx.x - warpSize; i < nWarpSizeAligned; i += blockDim.x - warpSize)
				dstChromosome[i] = srcChromosome[i];
		}
	}

	template <typename gene>
	__global__ void migrationKernel(gene* population, unsigned int islandPopulationSize, unsigned int nWarpSizeAligned, 
		unsigned int *cycleWeight, unsigned int* islandBest, unsigned int* islandWorst, bool *sourceInSecondBuffer) 
	{
		__shared__ gene *s_srcDst[2];

		unsigned int thisIsland = blockIdx.x;
		unsigned int nextIsland = (blockIdx.x + 1) % gridDim.x;

		if (threadIdx.x == 0) {
			unsigned int nextIslandBestIndex = islandBest[nextIsland];
			unsigned int thisIslandWorstIndex = islandWorst[thisIsland];

			unsigned int nextIslandBestIndexGlobal = nextIsland * islandPopulationSize + nextIslandBestIndex;
			unsigned int thisIslandWorstIndexGlobal = thisIsland * islandPopulationSize + thisIslandWorstIndex;

			cycleWeight[thisIslandWorstIndexGlobal] = cycleWeight[nextIslandBestIndexGlobal];

			bool nextSourceInSecondBuffer = sourceInSecondBuffer[nextIsland];
			bool thisSourceInSecondBuffer = sourceInSecondBuffer[thisIsland];

			s_srcDst[0] = population + nWarpSizeAligned * 
				(2 * nextIsland * islandPopulationSize + (nextSourceInSecondBuffer ? islandPopulationSize : 0) + nextIslandBestIndex);
			s_srcDst[1] = population + nWarpSizeAligned * 
				(2 * thisIsland * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + thisIslandWorstIndex);
		}

		__syncthreads();

		// Replace worst chromosome in thisIsland with best chromosome from nextIsland
		gene* srcChromosome = s_srcDst[0];
		gene* dstChromosome = s_srcDst[1];

		for (unsigned int i = threadIdx.x; i < nWarpSizeAligned; i += blockDim.x)
			dstChromosome[i] = srcChromosome[i];
	}

	template <typename Instance, typename gene>
	__global__ void islandEvolutionKernel(const Instance instance, curandState* globalState, gene* population, unsigned int islandPopulationSize, 
		unsigned int iterationCount, bool elitism, float crossoverProbability, float mutationProbability, 
		unsigned int* cycleWeight, unsigned int *islandBest, unsigned int *islandWorst, bool *sourceInSecondBuffer, unsigned int stalledIsolatedIterationsLimit) 
	{
		extern __shared__ unsigned int s_buffer[];
		unsigned int* s_reductionBuffer = s_buffer;
		unsigned int *s_cycleWeight = s_buffer + 4 * WARP_SIZE;
		unsigned int* s_islandBestIndex = s_cycleWeight + islandPopulationSize;
		unsigned int* s_islandWorstIndex = s_cycleWeight + islandPopulationSize + 1;
		float *s_roulletteWheelThreshold = (float*)(s_cycleWeight + islandPopulationSize + 2);

		unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;	// Global thread id
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);			// Warp thread id
		unsigned int nWarpSizeAligned = (size(instance) & ~(WARP_SIZE - 1)) + WARP_SIZE;
		bool thisSourceInSecondBuffer = sourceInSecondBuffer[blockIdx.x];
		unsigned int islandPrevBestCycleWeight = (unsigned int)-1, stalledIsolatedIterationsCounter = 0;

		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x)
			s_cycleWeight[chromosomeIndex] = cycleWeight[blockIdx.x * islandPopulationSize + chromosomeIndex];

		findMinMax(s_cycleWeight, islandPopulationSize, s_reductionBuffer, threadIdx.x, threadIdx.x, s_islandBestIndex, s_islandWorstIndex);

		__syncthreads();

		while (iterationCount-- > 0 && stalledIsolatedIterationsCounter < stalledIsolatedIterationsLimit) {

			unsigned int islandBestIndex = *s_islandBestIndex;
			unsigned int islandWorstIndex = *s_islandWorstIndex;

			// Selection
			{
				float maxThreshold = createRoulletteWheel(islandBestIndex, islandWorstIndex, islandPopulationSize, s_cycleWeight, s_roulletteWheelThreshold);

				for (unsigned int chromosomeIndex = threadIdx.x / WARP_SIZE; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x / WARP_SIZE) {

					// Select lane id based on roullette wheel selection
					unsigned int selectedIndex = elitism && chromosomeIndex == islandBestIndex ? chromosomeIndex : 
						selectIndex(globalState + tid, s_roulletteWheelThreshold, maxThreshold, islandPopulationSize);

					// Copy selected chromosome
					gene* srcChromosome = population + nWarpSizeAligned *
						(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + selectedIndex);
					gene* dstChromosome = population + nWarpSizeAligned *
						(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? 0 : islandPopulationSize) + chromosomeIndex);

					for (unsigned int i = lid; i < nWarpSizeAligned; i += WARP_SIZE)
						dstChromosome[i] = srcChromosome[i];
				}

				thisSourceInSecondBuffer = !thisSourceInSecondBuffer;
			}

			__syncthreads();

			for (unsigned int chromosomeIndex = 2 * (threadIdx.x / WARP_SIZE); chromosomeIndex < islandPopulationSize; chromosomeIndex += 2 * blockDim.x / WARP_SIZE) {
				gene* chromosomeA = population + nWarpSizeAligned * 
					(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + chromosomeIndex);
				gene* chromosomeB = population + nWarpSizeAligned * 
					(blockIdx.x * 2 * islandPopulationSize + (thisSourceInSecondBuffer ? islandPopulationSize : 0) + chromosomeIndex + 1);
				bool performCrossover, performMutation;
				unsigned int thisCycleWeight;

				// Crossover : chromosomeA x chromosomeB
				if (lid == 0) performCrossover = chromosomeIndex + 1 < islandPopulationSize && 
					(!elitism || (chromosomeIndex != islandBestIndex && chromosomeIndex + 1 != islandBestIndex)) && 
					crossoverProbability > curand_uniform(globalState + tid);
				if (__shfl_sync(FULL_MASK, performCrossover, 0)) {
					warpPMXCoalescedFix(chromosomeA, chromosomeB, size(instance), globalState + tid);
				}

				// Mutation : chromosomeA
				if (lid == 0) performMutation = (!elitism || chromosomeIndex != islandBestIndex) && 
					mutationProbability > curand_uniform(globalState + tid);
				if (__shfl_sync(FULL_MASK, performMutation, 0)) {
					warpIntervalMutate(chromosomeA, size(instance), globalState + tid);
				}

				// Fitness : chromosomeA
				thisCycleWeight = warpCalculateCycleWeight(chromosomeA, instance);
				if (lid == 0) s_cycleWeight[chromosomeIndex] = thisCycleWeight;

				if (chromosomeIndex + 1 < islandPopulationSize) {
					// Mutation : chromosomeB
					if (lid == 0) performMutation = (!elitism || chromosomeIndex + 1 != islandBestIndex) && 
						mutationProbability > curand_uniform(globalState + tid);
					if (__shfl_sync(FULL_MASK, performMutation, 0)) {
						warpIntervalMutate(chromosomeB, size(instance), globalState + tid);
					}

					// Fitness : chromosomeB
					thisCycleWeight = warpCalculateCycleWeight(chromosomeB, instance);
					if (lid == 0) s_cycleWeight[chromosomeIndex + 1] = thisCycleWeight;
				}
			}

			__syncthreads();

			// Best and Worst
			findMinMax(s_cycleWeight, islandPopulationSize, s_reductionBuffer, threadIdx.x, threadIdx.x, s_islandBestIndex, s_islandWorstIndex);

			__syncthreads();

			unsigned int islandCurrBestCycleWeight = s_cycleWeight[*s_islandBestIndex];
			if (islandPrevBestCycleWeight == islandCurrBestCycleWeight) {
				stalledIsolatedIterationsCounter++;
			} else {
				stalledIsolatedIterationsCounter = 0;
			}
			islandPrevBestCycleWeight = islandCurrBestCycleWeight;
		}

		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < islandPopulationSize; chromosomeIndex += blockDim.x)
			cycleWeight[blockIdx.x * islandPopulationSize + chromosomeIndex] = s_cycleWeight[chromosomeIndex];

		if (threadIdx.x == 0) {
			if (*s_islandBestIndex == *s_islandWorstIndex) {
				islandBest[blockIdx.x] = 0;
				islandWorst[blockIdx.x] = islandPopulationSize - 1;
			} else {
				islandBest[blockIdx.x] = *s_islandBestIndex;
				islandWorst[blockIdx.x] = *s_islandWorstIndex;
			}
			sourceInSecondBuffer[blockIdx.x] = thisSourceInSecondBuffer;
		}
	}

	template <typename Instance, typename gene = unsigned short>
	int solveTSPFineGrained(const Instance instance, struct IslandGeneticAlgorithmOptions options, gene *continentBestCycle, int blockWarpCount, const int mpiRank, const int mpiSize, MPI_Datatype mpiGene, int seed, const std::string& historyPathname, bool reportProgress = false)
	{
		cudaError status;
		unsigned int nWarpSizeAligned = (size(instance) & ~(WARP_SIZE - 1)) + WARP_SIZE;
		unsigned int* d_cycleWeight, * d_islandBest, * d_islandWorst;
		bool *d_sourceInSecondBuffer;
		gene* d_population;
		curandState* d_globalState;
		unsigned int* h_cycleWeight = new unsigned int[options.islandCount * options.islandPopulationSize];
		unsigned int* h_islandBest = new unsigned int[options.islandCount];
		unsigned int* h_islandWorst = new unsigned int[options.islandCount];
		unsigned int stalledMigrationsCount = 0, stalledBestCycleWeight = std::numeric_limits<unsigned int>::max();
		unsigned int continentBestCycleWeight = std::numeric_limits<unsigned int>::max();
		bool immigrationOngoing = false;
		gene *d_immigrationBuffer, *d_emigrationBuffer;
		MPI_Request immigrationRequest, emigrationRequest;

		if ((status = cudaMalloc(&d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(unsigned int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_population, 2 * nWarpSizeAligned * options.islandCount * options.islandPopulationSize * sizeof(gene))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_globalState, options.islandCount * options.islandPopulationSize * WARP_SIZE * sizeof(curandState))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_islandBest, options.islandCount * sizeof(unsigned int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_islandWorst, options.islandCount * sizeof(unsigned int))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_sourceInSecondBuffer, options.islandCount * sizeof(bool))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_immigrationBuffer, nWarpSizeAligned * options.islandCount * sizeof(gene))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMalloc(&d_emigrationBuffer, nWarpSizeAligned * options.islandCount * sizeof(gene))) != cudaSuccess) {
			std::cerr << "Could not allocate device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		setupCurand<<<options.islandCount, blockWarpCount * WARP_SIZE>>>(d_globalState, seed);

		if ((status = cudaGetLastError()) != cudaSuccess) {
			std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		initializationKernel<<<options.islandCount, blockWarpCount * WARP_SIZE, (options.islandPopulationSize + 4 * WARP_SIZE) * sizeof(unsigned int)>>>(
			instance, d_globalState, d_population, options.islandPopulationSize, d_cycleWeight, d_islandBest, d_islandWorst
		);

		if ((status = cudaGetLastError()) != cudaSuccess) {
			std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMemset(d_sourceInSecondBuffer, false, options.islandCount * sizeof(bool))) != cudaSuccess) {
			std::cerr << "Could not set device memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMemcpy(h_cycleWeight, d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaMemcpy(h_islandBest, d_islandBest, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if (reportProgress && (status = cudaMemcpy(h_islandWorst, d_islandWorst, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
			goto FREE;
		}

		updateStalledMigrationsCount(stalledMigrationsCount, stalledBestCycleWeight, h_cycleWeight, h_islandBest, options.islandCount, options.islandPopulationSize);

		if (reportProgress) {
			std::cout << "[" << mpiRank << "] INITIAL (stable streak: " << stalledMigrationsCount << ")" << std::endl <<
				std::setw(8) << std::left << "Best:";
			for (unsigned int i = 0; i < options.islandCount; i++)
				std::cout << std::setw(12) << std::right << h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]];
			std::cout << std::endl << std::setw(8) << std::left << "Worst:";
			for (unsigned int i = 0; i < options.islandCount; i++)
				std::cout << std::setw(12) << std::right << h_cycleWeight[i * options.islandPopulationSize + h_islandWorst[i]];
			std::cout << std::endl;
		}

		for (unsigned int migrationNumber = 1; migrationNumber <= options.migrationCount && stalledMigrationsCount < options.stalledMigrationsLimit; migrationNumber++) {

			if (migrationNumber % options.intercontinentalMigrationPeriod == 0) {

				if (migrationNumber > options.intercontinentalMigrationPeriod) {
					int emigrationComplete;
					MPI_Test(&emigrationRequest, &emigrationComplete, MPI_STATUS_IGNORE);
					if (!emigrationComplete) {
						MPI_Cancel(&emigrationRequest);
						MPI_Wait(&emigrationRequest, MPI_STATUS_IGNORE);
					}
				}

				emigrationKernel<<<options.islandCount, blockWarpCount * WARP_SIZE>>>(
					d_population, options.islandPopulationSize, nWarpSizeAligned,
					d_islandBest, d_sourceInSecondBuffer, d_emigrationBuffer
				);

				if ((status = cudaGetLastError()) != cudaSuccess) {
					std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}

				if (reportProgress)
					std::cout << "EMIGRATION on " << migrationNumber << " [" << mpiRank << " -> " << (mpiRank + 1) % mpiSize << "]" << std::endl;

				if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
					std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}

				MPI_Isend(d_emigrationBuffer, nWarpSizeAligned * options.islandCount, mpiGene,
					(mpiRank + 1) % mpiSize, 1, MPI_COMM_WORLD, &emigrationRequest);

				if (!immigrationOngoing) {
					MPI_Irecv(d_immigrationBuffer, nWarpSizeAligned * options.islandCount, mpiGene,
						(mpiRank + mpiSize - 1) % mpiSize, 1, MPI_COMM_WORLD, &immigrationRequest);
					immigrationOngoing = true;
				}
			}

			int immigrationComplete = 0;
			if (immigrationOngoing) {
				MPI_Test(&immigrationRequest, &immigrationComplete, MPI_STATUS_IGNORE);
				if (immigrationComplete) {
					immigrationOngoing = false;
					if (reportProgress)
						std::cout << "IMMIGRATION on " << migrationNumber << " [" << (mpiRank + mpiSize - 1) % mpiSize << "] -> [" << mpiRank << "]" << std::endl;

					immigrationKernel<<<options.islandCount, blockWarpCount * WARP_SIZE>>>(
						instance, d_population, options.islandPopulationSize, nWarpSizeAligned,
						d_cycleWeight, d_islandWorst, d_sourceInSecondBuffer, d_immigrationBuffer
					);

					if ((status = cudaGetLastError()) != cudaSuccess) {
						std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
						goto FREE;
					}

					if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
						std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
						goto FREE;
					}
				}
			}

			if (!immigrationComplete) {
				migrationKernel<<<options.islandCount, blockWarpCount * WARP_SIZE>>>(
					d_population, options.islandPopulationSize, nWarpSizeAligned,
					d_cycleWeight, d_islandBest, d_islandWorst, d_sourceInSecondBuffer
				);

				if ((status = cudaGetLastError()) != cudaSuccess) {
					std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}
			}

			islandEvolutionKernel<<<options.islandCount, blockWarpCount * WARP_SIZE, (options.islandPopulationSize + 4 * WARP_SIZE + 2) * sizeof(unsigned int) + (options.islandPopulationSize + WARP_SIZE) * sizeof(float)>>>(
				instance, d_globalState, d_population, options.islandPopulationSize,
				options.isolatedIterationCount, options.elitism, options.crossoverProbability, options.mutationProbability,
				d_cycleWeight, d_islandBest, d_islandWorst, d_sourceInSecondBuffer, options.stalledIsolatedIterationsLimit
			);

			if ((status = cudaGetLastError()) != cudaSuccess) {
				std::cerr << "Could not launch kernel: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaMemcpy(h_cycleWeight, d_cycleWeight, options.islandCount * options.islandPopulationSize * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaMemcpy(h_islandBest, d_islandBest, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if (reportProgress && (status = cudaMemcpy(h_islandWorst, d_islandWorst, options.islandCount * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if (!historyPathname.empty()) {
				continentBestCycleWeight = std::numeric_limits<unsigned int>::max();
				unsigned int continentBestIslandIndex = std::numeric_limits<unsigned int>::max();
				bool continentBestIslandSourceInSecondBuffer = false;
				for (unsigned int i = 0; i < options.islandCount; i++) {
					if (continentBestCycleWeight > h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]]) {
						continentBestCycleWeight = h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]];
						continentBestIslandIndex = i;
					}
				}

				if ((status = cudaMemcpy(&continentBestIslandSourceInSecondBuffer, d_sourceInSecondBuffer + continentBestIslandIndex, sizeof(bool), cudaMemcpyDeviceToHost)) != cudaSuccess) {
					std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}

				if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
					std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}

				gene *d_continentBestCycle = d_population + nWarpSizeAligned *
					(continentBestIslandIndex * 2 * options.islandPopulationSize +
						(continentBestIslandSourceInSecondBuffer ? options.islandPopulationSize : 0) +
					h_islandBest[continentBestIslandIndex]);

				if ((status = cudaMemcpy(continentBestCycle, d_continentBestCycle, size(instance) * sizeof(gene), cudaMemcpyDeviceToHost)) != cudaSuccess) {
					std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}
			}

			updateStalledMigrationsCount(stalledMigrationsCount, stalledBestCycleWeight, h_cycleWeight, h_islandBest, options.islandCount, options.islandPopulationSize);

			if (reportProgress) {
				std::cout << "[" << mpiRank << "] " "CYCLE: " << migrationNumber <<
					" (stable streak: " << stalledMigrationsCount << ")" << std::endl <<
					std::setw(8) << std::left << "Best:";
				for (unsigned int i = 0; i < options.islandCount; i++)
					std::cout << std::setw(12) << std::right << h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]];
				std::cout << "\n" << std::setw(8) << std::left << "Worst:";
				for (unsigned int i = 0; i < options.islandCount; i++)
					std::cout << std::setw(12) << std::right << h_cycleWeight[i * options.islandPopulationSize + h_islandWorst[i]];
				std::cout << std::endl;
			}

			if (!historyPathname.empty()) {

				if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
					std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
					goto FREE;
				}

				std::ostringstream historyFullPathnameStream;
				historyFullPathnameStream << historyPathname <<
					std::setw(3) << std::setfill('0') << mpiRank << "_" <<
					std::setw(4) << std::setfill('0') << migrationNumber << ".json";
				std::string historyFullPathname = historyFullPathnameStream.str();
				std::ofstream history(historyFullPathname);
				if (history.is_open()) {
					auto cycleJson = nlohmann::json::array();
					cycleJson.get_ptr<nlohmann::json::array_t*>()->reserve(size(instance));
					for (int i = 0; i < size(instance); i++)
						cycleJson.emplace_back(continentBestCycle[i] + 1);

					auto cycleWeightsJson = nlohmann::json::array();
					cycleWeightsJson.get_ptr<nlohmann::json::array_t*>()->reserve(options.islandCount * options.islandPopulationSize);
					for (int i = 0; i < options.islandCount * options.islandPopulationSize; i++)
						cycleWeightsJson.emplace_back(h_cycleWeight[i]);

					nlohmann::json iterationJson = {
						{ "migration_number", migrationNumber },
						{ "best_distance", continentBestCycleWeight },
						{ "best_path", cycleJson },
						{ "population_heatmap", cycleWeightsJson }
					};
					history << iterationJson.dump(-1);
					history.close();
				} else {
					std::cerr << "Could not open file " << historyFullPathname << "\n";
				}
			}
		}

		{
			continentBestCycleWeight = std::numeric_limits<unsigned int>::max();
			unsigned int globalBestIslandIndex = std::numeric_limits<unsigned int>::max();
			bool globalBestIslandSourceInSecondBuffer = false;
			for (unsigned int i = 0; i < options.islandCount; i++) {
				if (continentBestCycleWeight > h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]]) {
					continentBestCycleWeight = h_cycleWeight[i * options.islandPopulationSize + h_islandBest[i]];
					globalBestIslandIndex = i;
				}
			}

			if ((status = cudaMemcpy(&globalBestIslandSourceInSecondBuffer, d_sourceInSecondBuffer + globalBestIslandIndex, sizeof(bool), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			gene *d_globalBestCycle = d_population + nWarpSizeAligned *
				(globalBestIslandIndex * 2 * options.islandPopulationSize +
					(globalBestIslandSourceInSecondBuffer ? options.islandPopulationSize : 0) +
				h_islandBest[globalBestIslandIndex]);

			if ((status = cudaMemcpy(continentBestCycle, d_globalBestCycle, size(instance) * sizeof(gene), cudaMemcpyDeviceToHost)) != cudaSuccess) {
				std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}

			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
				goto FREE;
			}
		}

FREE:
		if (immigrationOngoing) {
			MPI_Cancel(&immigrationRequest);
			MPI_Wait(&immigrationRequest, MPI_STATUS_IGNORE);
		}

		if (options.migrationCount > options.intercontinentalMigrationPeriod) {
			int emigrationComplete;
			MPI_Test(&emigrationRequest, &emigrationComplete, MPI_STATUS_IGNORE);
			if (!emigrationComplete) {
				MPI_Cancel(&emigrationRequest);
				MPI_Wait(&emigrationRequest, MPI_STATUS_IGNORE);
			}
		}

		cudaFree(d_cycleWeight);
		cudaFree(d_population);
		cudaFree(d_globalState);
		cudaFree(d_islandBest);
		cudaFree(d_islandWorst);
		cudaFree(d_sourceInSecondBuffer);
		cudaFree(d_immigrationBuffer);
		cudaFree(d_emigrationBuffer);

		delete[] h_cycleWeight;
		delete[] h_islandBest;
		delete[] h_islandWorst;

		return continentBestCycleWeight;
	}

}

#endif
