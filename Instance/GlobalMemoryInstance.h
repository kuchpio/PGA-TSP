#ifndef __GLOBAL_MEMORY_INSTANCE_H__
#define __GLOBAL_MEMORY_INSTANCE_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Helper.h"

namespace tsp {

	typedef struct GlobalMemoryInstance {
		int* d_adjecencyMatrix = NULL;
		int size = 0;
	} GlobalMemoryInstance;

	template <typename Metric>
	bool initInstance(GlobalMemoryInstance* instance, const float* x, const float* y, const int size, Metric metric) {
		float* d_x, * d_y;
		int* d_adjecencyMatrix;

		if (cudaMalloc(&d_x, size * sizeof(float)) != cudaSuccess)
			return false;
		if (cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return false;
		if (cudaMalloc(&d_y, size * sizeof(float)) != cudaSuccess)
			return false;
		if (cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return false;
		if (cudaMalloc(&d_adjecencyMatrix, size * size * sizeof(int)) != cudaSuccess)
			return false;

		fillAdjecencyMatrixKernel << <4, 256 >> > (d_adjecencyMatrix, d_x, d_y, size, metric);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return false;

		cudaFree(d_x);
		cudaFree(d_y);

		instance->d_adjecencyMatrix = d_adjecencyMatrix;
		instance->size = size;

		return true;
	}

	void destroyInstance(GlobalMemoryInstance instance) {
		cudaFree((int*)instance.d_adjecencyMatrix);
	}

	__device__ __host__
		int size(const GlobalMemoryInstance instance) {
		return instance.size;
	}

	__device__
		int edgeWeight(const GlobalMemoryInstance instance, const int from, const int to) {
		return instance.d_adjecencyMatrix[from * size(instance) + to];
	}

}

#endif
