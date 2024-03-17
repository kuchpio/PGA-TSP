#include "GlobalMemoryInstance.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ 
void fillAdjecencyMatrixRow(const float *x, const float *y, const IMetric *metric, const int size, int *adjecencyMatrix) {
    int tid, row, col;
    for (tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size * size; tid += blockDim.x * gridDim.x) {
        row = tid / size;
        col = tid % size;
        adjecencyMatrix[tid] = metric->distance(x[row], y[row], x[col], y[col]);
    }
}

GlobalMemoryInstance::GlobalMemoryInstance(const float *x, const float *y, const int size, const IMetric *metric): _size(size) {
    cudaError_t status; 

    if ((status = cudaMalloc(&this->d_adjecencyMatrix, size * size * sizeof(int))) != cudaSuccess) {
        std::cerr << "Could not allocate device adjecency matrix (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }

    // Run kernel that fills adjecency matrix
    fillAdjecencyMatrixRow<<<4, 256>>>(x, y, metric, size, this->d_adjecencyMatrix);

    if ((status = cudaGetLastError()) != cudaSuccess) {
        std::cerr << "Could not fill adjecency matrix on device (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }

    if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
        std::cerr << "Could not synchronize device (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }
}

int GlobalMemoryInstance::size() const {
#ifdef __CUDA_ARCH__
    return this->_size;
#else
    return -1;
#endif
}

int GlobalMemoryInstance::edgeWeight(const int from, const int to) const {
#ifdef __CUDA_ARCH__
    return this->d_adjecencyMatrix[from * this->_size + to];
#else
    return -1;
#endif
}

int GlobalMemoryInstance::hamiltonianCycleWeight(const int *cycle) const {
#ifdef __CUDA_ARCH__
    return this->_size;
    int sum = this->edgeWeight(this->_size - 1, 0);

    for (int i = 0; i < this->_size - 1; i++) {
        sum += this->edgeWeight(i, i + 1);
    }

    return sum;
#else
    return -1;
#endif
}

GlobalMemoryInstance::~GlobalMemoryInstance() {
    cudaFree(this->d_adjecencyMatrix);
}
