#ifndef __HELPER_H__
#define __HELPER_H__

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <limits>
#include <optional>
#include <mpi.h>

namespace tsp {

	struct IslandGeneticAlgorithmOptions {
		unsigned int islandCount;
		unsigned int islandPopulationSize;
		unsigned int isolatedIterationCount;
		unsigned int migrationCount;
		unsigned int superemigrationPeriod;
		float crossoverProbability;
		float mutationProbability;
		bool elitism;
		unsigned int stalledIsolatedIterationsLimit;
		unsigned int stalledMigrationsLimit;
	};

	__global__
	void setupCurand(curandState* globalState, int seed) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(seed, tid, 0, globalState + tid);
	}

	template <bool reduceMin = true, bool reduceMax = true>
	__device__ __forceinline__ void findMinMax(const unsigned int* array, const unsigned int arraySize, unsigned int* reductionBuffer,
		unsigned int minIndex, unsigned int maxIndex, unsigned int *minIndexOutput, unsigned int *maxIndexOutput) 
	{
		unsigned int wid = threadIdx.x / WARP_SIZE;			// Block warp id
		unsigned int lid = threadIdx.x & (WARP_SIZE - 1);	// Warp thread id

		unsigned int* reducedMin, * reducedMinIndex, * reducedMax, * reducedMaxIndex;
		unsigned int min, max;

		if (reduceMin) {
			reducedMin = reductionBuffer;
			reducedMinIndex = reducedMin + WARP_SIZE;
			min = 0xffffffff;
			minIndex = threadIdx.x;
		}
		if (reduceMax) {
			reducedMax = reductionBuffer + (reduceMin ? 2 * WARP_SIZE : 0);
			reducedMaxIndex = reducedMax + WARP_SIZE;
			max = 0;
			maxIndex = threadIdx.x;
		}

		// 1. Block stride loop thread reduction
		for (unsigned int chromosomeIndex = threadIdx.x; chromosomeIndex < arraySize; chromosomeIndex += blockDim.x) {
			unsigned int value = array[chromosomeIndex];
			if (reduceMin && min > value) {
				min = value;
				minIndex = chromosomeIndex;
			}
			if (reduceMax && max < value) {
				max = value;
				maxIndex = chromosomeIndex;
			}
		}

		// 2. Warp reduction in each warp
		unsigned int minShuf, minIndexShuf, maxShuf, maxIndexShuf;
		for (int i = 1; i < WARP_SIZE; i *= 2) {

			if (reduceMin) {
				minShuf = __shfl_xor_sync(FULL_MASK, min, i);
				minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
				if (min > minShuf) {
					minIndex = minIndexShuf;
					min = minShuf;
				}
			}

			if (reduceMax) {
				maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
				maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
				if (max < maxShuf) {
					maxIndex = maxIndexShuf;
					max = maxShuf;
				}
			}

		}

		if (lid == 0) {
			if (reduceMin) {
				reducedMin[wid] = min;
				reducedMinIndex[wid] = minIndex;
			}
			if (reduceMax) {
				reducedMax[wid] = max;
				reducedMaxIndex[wid] = maxIndex;
			}
		}

		__syncthreads();

		// 3. Warp reduction of reduced results
		if (wid == 0) {

			if (reduceMin) {
				min = reducedMin[lid];
				minIndex = reducedMinIndex[lid];
				if (lid >= blockDim.x / WARP_SIZE) min = 0xffffffff;
			}
			if (reduceMax) {
				max = reducedMax[lid];
				maxIndex = reducedMaxIndex[lid];
				if (lid >= blockDim.x / WARP_SIZE) max = 0;
			}

			for (int i = 1; i < WARP_SIZE; i *= 2) {

				if (reduceMin) {
					minShuf = __shfl_xor_sync(FULL_MASK, min, i);
					minIndexShuf = __shfl_xor_sync(FULL_MASK, minIndex, i);
					if (min > minShuf) {
						minIndex = minIndexShuf;
						min = minShuf;
					}
				}

				if (reduceMax) {
					maxShuf = __shfl_xor_sync(FULL_MASK, max, i);
					maxIndexShuf = __shfl_xor_sync(FULL_MASK, maxIndex, i);
					if (max < maxShuf) {
						maxIndex = maxIndexShuf;
						max = maxShuf;
					}
				}

			}

			if (lid == 0) {
				if (minIndex == maxIndex) {
					*minIndexOutput = 0;
					*maxIndexOutput = arraySize - 1;
				} else {
					*minIndexOutput = minIndex;
					*maxIndexOutput = maxIndex;
				}
			}
		}
	}

	inline void updateStalledMigrationsCount(unsigned int &stalledMigrationsCount, unsigned int &stalledBestCycleWeight,
		const unsigned int *h_cycleWeight, const unsigned int islandPopulationSize, const unsigned int *h_islandBest, const unsigned int islandCount)
	{
		bool stable = true;
		const unsigned int firstBestCycleWeight = h_cycleWeight[h_islandBest[0]];
		for (unsigned int i = 1; i < islandCount; i++) {
			if (firstBestCycleWeight != h_cycleWeight[i * islandPopulationSize + h_islandBest[i]]) {
				stable = false;
				break;
			}
		}
		if (stable) {
			if (firstBestCycleWeight != stalledBestCycleWeight) {
				stalledBestCycleWeight = firstBestCycleWeight;
				stalledMigrationsCount = 0;
			}
			stalledMigrationsCount++;
		} else {
			stalledBestCycleWeight = (unsigned int)-1;
			stalledMigrationsCount = 0;
		}
	}

	template<typename gene>
	bool verifyResults(const tsp::IHostInstance* instance, gene* bestCycle, unsigned int bestCycleWeight)
	{
		const unsigned int n = instance->size();
		std::vector<bool> visited(n, false);
		unsigned int verifiedCycleWeight = instance->edgeWeight(bestCycle[n - 1], bestCycle[0]);
		visited[bestCycle[n - 1]] = true;

		for (unsigned int i = 0; i < n - 1; i++) {
			if (visited[bestCycle[i]]) {
				std::cerr << "VERIFICATION: Cycle is not hamiltonian. Vertex " << bestCycle[i] << " repeated.\n";
				return false;
			}
			verifiedCycleWeight += instance->edgeWeight(bestCycle[i], bestCycle[i + 1]);
			visited[bestCycle[i]] = true;
		}

		if (bestCycleWeight != verifiedCycleWeight) {
			std::cerr << "VERIFICATION: Cycle has different length (" << verifiedCycleWeight << ") than returned best cycle length (" << bestCycleWeight << ").\n";
			return false;
		}

		return true;
	}

	inline bool isThresholdAchieved(
		const unsigned int* h_cycleWeight, const unsigned int islandPopulationSize,
		const unsigned int* h_islandBest, const unsigned int islandCount,
		const unsigned int threshold
		) {

		for (unsigned int i = 0; i < islandCount; i++) {
			if (threshold >= h_cycleWeight[i * islandPopulationSize + h_islandBest[i]]) {
				return true;
			}
		}

		return false;
	}

	template<typename gene>
	std::optional<unsigned int> computeBestCycle(
		const gene* d_population, const unsigned int* h_cycleWeight, const unsigned int islandPopulationSize,
		const unsigned int* h_islandBest, const bool *d_sourceInSecondBuffer, const unsigned int islandCount,
		gene *h_continentBestCycle, const unsigned int chromosomeSize
		) {

		cudaError status;
		const unsigned int nWarpSizeAligned = (chromosomeSize & ~(WARP_SIZE - 1)) + WARP_SIZE;
		unsigned int continentBestCycleWeight = std::numeric_limits<unsigned int>::max();
		unsigned int continentBestIslandIndex = std::numeric_limits<unsigned int>::max();
		bool continentBestIslandSourceInSecondBuffer = false;
		for (unsigned int i = 0; i < islandCount; i++) {
			if (continentBestCycleWeight > h_cycleWeight[i * islandPopulationSize + h_islandBest[i]]) {
				continentBestCycleWeight = h_cycleWeight[i * islandPopulationSize + h_islandBest[i]];
				continentBestIslandIndex = i;
			}
		}

		if ((status = cudaMemcpy(&continentBestIslandSourceInSecondBuffer, d_sourceInSecondBuffer + continentBestIslandIndex, sizeof(bool), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
			return std::nullopt;
		}

		if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
			std::cerr << "Could not synchronize device: " << cudaGetErrorString(status) << ".\n";
			return std::nullopt;
		}

		const gene *d_continentBestCycle = d_population + nWarpSizeAligned *
			(continentBestIslandIndex * 2 * islandPopulationSize +
				(continentBestIslandSourceInSecondBuffer ? islandPopulationSize : 0) +
			h_islandBest[continentBestIslandIndex]);

		if ((status = cudaMemcpy(h_continentBestCycle, d_continentBestCycle, chromosomeSize * sizeof(gene), cudaMemcpyDeviceToHost)) != cudaSuccess) {
			std::cerr << "Could not copy device memory to host memory: " << cudaGetErrorString(status) << ".\n";
			return std::nullopt;
		}

		return continentBestCycleWeight;
	}

	template<typename gene>
	void saveCurrentIteration(
		const std::string& historyPathname, const int mpiRank, const unsigned int migrationNumber,
		unsigned int continentBestCycleWeight, const gene* continentBestCycle, const unsigned int chromosomeSize,
		unsigned int* cycleWeight, unsigned int populationSize
		) {
		std::ostringstream historyFullPathnameStream;
		historyFullPathnameStream << historyPathname <<
			std::setw(3) << std::setfill('0') << mpiRank << "_" <<
			std::setw(4) << std::setfill('0') << migrationNumber << ".json";
		std::string historyFullPathname = historyFullPathnameStream.str();
		std::ofstream history(historyFullPathname);
		if (history.is_open()) {
			auto cycleJson = nlohmann::json::array();
			cycleJson.get_ptr<nlohmann::json::array_t*>()->reserve(chromosomeSize);
			for (unsigned int i = 0; i < chromosomeSize; i++)
				cycleJson.emplace_back(continentBestCycle[i] + 1);

			auto cycleWeightsJson = nlohmann::json::array();
			cycleWeightsJson.get_ptr<nlohmann::json::array_t*>()->reserve(populationSize);
			for (unsigned int i = 0; i < populationSize; i++)
				cycleWeightsJson.emplace_back(cycleWeight[i]);

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

	inline void printIterationStats(const int mpiRank, const unsigned int migrationNumber, const unsigned int stalledMigrationsCount,
		const unsigned int* h_cycleWeight, const unsigned int islandPopulationSize,
		const unsigned int* h_islandBest, const unsigned int* h_islandWorst, const unsigned int islandCount
		) {
		std::cout << "[" << mpiRank << "] " "CYCLE: " << migrationNumber <<
			" (stable streak: " << stalledMigrationsCount << ")" << std::endl <<
			std::setw(8) << std::left << "Best:";
		for (unsigned int i = 0; i < islandCount; i++)
			std::cout << std::setw(12) << std::right << h_cycleWeight[i * islandPopulationSize + h_islandBest[i]];
		std::cout << "\n" << std::setw(8) << std::left << "Worst:";
		for (unsigned int i = 0; i < islandCount; i++)
			std::cout << std::setw(12) << std::right << h_cycleWeight[i * islandPopulationSize + h_islandWorst[i]];
		std::cout << std::endl;
	}

	inline std::optional<int> getMinThresholdDurationMs(const std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>> thresholdTime,
		const int mpiRank, const int mpiSize, const std::chrono::time_point<std::chrono::high_resolution_clock> startTime) {

		int minThresholdDurationMs = -1;
		MPI_Comm belowThresholdComm;
		int belowThresholdCommRank, belowThresholdCommSize;
		MPI_Comm_split(MPI_COMM_WORLD, thresholdTime.has_value(), mpiRank, &belowThresholdComm);
		MPI_Comm_rank(belowThresholdComm, &belowThresholdCommRank);
		MPI_Comm_size(belowThresholdComm, &belowThresholdCommSize);
		if (thresholdTime.has_value()) {
			const int thresholdDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(thresholdTime.value() - startTime).count();
			MPI_Reduce(&thresholdDurationMs, &minThresholdDurationMs, 1, MPI_INT, MPI_MIN, 0, belowThresholdComm);
			if (belowThresholdCommRank == 0 && mpiRank != 0)
				MPI_Send(&minThresholdDurationMs, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
		}
		MPI_Comm_free(&belowThresholdComm);

		if (mpiRank == 0 && !thresholdTime.has_value() && belowThresholdCommSize < mpiSize)
			MPI_Recv(&minThresholdDurationMs, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (minThresholdDurationMs == -1) return std::nullopt;
		return minThresholdDurationMs;
	}

}

#endif
