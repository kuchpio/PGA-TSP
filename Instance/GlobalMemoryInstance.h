#include "DeviceInstance.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

template<class Metric>
__global__
void fillAdjecencyMatrixKernel(int* __restrict__ adjecencyMatrix, const float* __restrict__ x, const float* __restrict__ y, const int size) {
	int tid, row, col;
	for (tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size * size; tid += blockDim.x * gridDim.x) {
		row = tid / size;
		col = tid % size;
		adjecencyMatrix[tid] = Metric::distance(x[row], y[row], x[col], y[col]);
	}
}

class GlobalMemoryInstance : DeviceInstance<GlobalMemoryInstance> {
    private:
        const int *_adjecencyMatrix;
        const int _size;

    public:
        __device__ GlobalMemoryInstance(const int* adjecencyMatrix, const int size) : _adjecencyMatrix(adjecencyMatrix), _size(size) { }

        static void initAdjecencyMatrix(int* adjecencyMatrix, const int size);

        template<class Metric>
        static void fillAdjecencyMatrix(int* __restrict__ adjecencyMatrix, const float* __restrict__ x, const float* __restrict__ y, const int size) {
            cudaError_t status;

            fillAdjecencyMatrixKernel<Metric><<<4, 256>>>(adjecencyMatrix, x, y, size);

			if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
				std::cerr << "Could not synchronize device (" << 
					cudaGetErrorString(status) << ").\n";
				return;
			}
        }

        __device__ int size() const;
        __device__ int edgeWeight(const int from, const int to) const;
        __device__ int hamiltonianCycleWeight(const int *cycle) const;
        __device__ ~GlobalMemoryInstance();
};
