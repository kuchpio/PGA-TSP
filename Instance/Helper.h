#ifndef __INSTANCE_HELPER_H__
#define __INSTANCE_HELPER_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace tsp {

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

}

#endif
