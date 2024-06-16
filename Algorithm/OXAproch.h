#ifndef __OX_APPROACH_H__
#define __OX_APPROACH_H__
namespace tsp {
	template <typename Instance>
	int solveTSPOXApproach(const Instance instance, struct IslandGeneticAlgorithmOptions options, int* globalBestCycle, int seed) {
		const int blockSize = 256, gridSize = options.islandCount;
		int* d_fitness, * d_population, opt = -1;
		curandState* d_globalState;

		if (cudaMalloc(&d_fitness, blockSize * gridSize * sizeof(int)) != cudaSuccess)
			goto FREE;

		if (cudaMalloc(&d_population, blockSize * gridSize * size(instance) * sizeof(int)) != cudaSuccess)
			goto FREE;

		if (cudaMalloc(&d_globalState, blockSize * gridSize * sizeof(curandState)) != cudaSuccess)
			goto FREE;

		setupCurand << <gridSize, blockSize >> > (d_globalState, seed);

		if (cudaDeviceSynchronize() != cudaSuccess)
			goto FREE;
		geneticOXAlgorithmKernel << <gridSize, blockSize >> > (instance, d_fitness, d_population, d_globalState, options.crossoverProbability, options.mutationProbability, options.isolatedIterationCount * options.migrationCount);

		if (cudaDeviceSynchronize() != cudaSuccess)
			goto FREE;

		thrust::device_ptr<int> thrust_d_fitness(d_fitness);
		auto max_iter = thrust::max_element(thrust_d_fitness, thrust_d_fitness + blockSize * gridSize);
		int max_index = max_iter - thrust_d_fitness;
		cudaMemcpy(globalBestCycle, d_population + max_index * size(instance), size(instance) * sizeof(int), cudaMemcpyDeviceToHost);
		opt = MAX_DISTANCE_CAN - *max_iter;

	FREE:
		cudaFree(d_fitness);
		cudaFree(d_population);
		cudaFree(d_globalState);

		return opt;
	}
}
#endif