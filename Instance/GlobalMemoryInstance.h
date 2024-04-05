#ifndef __GLOBAL_MEMORY_INSTANCE_H__
#define __GLOBAL_MEMORY_INSTANCE_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace tsp {

	typedef struct GlobalMemoryInstance {
		const int* d_adjecencyMatrix = NULL;
		const int size = 0;
	} GlobalMemoryInstance;

	template<class Metric>
	__global__
	void fillAdjecencyMatrixKernel(int* adjecencyMatrix, const float* x, const float* y, const int size, Metric metric) {
		int tid, row, col;
		for (tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size * size; tid += blockDim.x * gridDim.x) {
			row = tid / size;
			col = tid % size;
			adjecencyMatrix[tid] = distance(metric, x[row], y[row], x[col], y[col]);
		}
	}

	template <typename Metric>
	const GlobalMemoryInstance initInstance(const float* x, const float* y, const int size, Metric metric) {
		float* d_x, * d_y;
		int* d_adjecencyMatrix;

		if (cudaMalloc(&d_x, size * sizeof(float)) != cudaSuccess)
			return { };
		if (cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return { };
		if (cudaMalloc(&d_y, size * sizeof(float)) != cudaSuccess)
			return { };
		if (cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return { };
		if (cudaMalloc(&d_adjecencyMatrix, size * size * sizeof(int)) != cudaSuccess)
			return { };

		fillAdjecencyMatrixKernel << <4, 256 >> > (d_adjecencyMatrix, d_x, d_y, size, metric);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return { };

		cudaFree(d_x);
		cudaFree(d_y);

		return { d_adjecencyMatrix, size };
	}

	bool isValid(GlobalMemoryInstance instance) {
		return instance.d_adjecencyMatrix != NULL;
	}

	void destroyInstance(GlobalMemoryInstance instance) {
		cudaFree((int*)instance.d_adjecencyMatrix);
	}

	__device__
		int size(const GlobalMemoryInstance instance) {
		return instance.size;
	}

	__device__
		int edgeWeight(const GlobalMemoryInstance instance, const int from, const int to) {
		return instance.d_adjecencyMatrix[from * size(instance) + to];
	}

	__device__
		int hamiltonianCycleWeight(const GlobalMemoryInstance instance, const int* cycle) {
		int sum = edgeWeight(instance, size(instance) - 1, 0);

		for (int i = 0; i < size(instance) - 1; i++) {
			sum += edgeWeight(instance, i, i + 1);
		}

		return sum;
	}

}

#endif
