#include "GlobalMemoryInstance.h"
#include <cuda_runtime.h>
#include <iostream>

GlobalMemoryInstance::GlobalMemoryInstance(float *x, float *y, const IMetric *metric, int size): _size(size) {
    cudaError_t status; 

    if ((status = cudaMalloc(&this->d_adjecencyMatrix, size * size * sizeof(int))) != cudaSuccess) {
        std::cerr << "Could not allocate device adjecency matrix (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }

    // Run kernel that fills adjecency matrix

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

int GlobalMemoryInstance::edgeWeight(int from, int to) const {
#ifdef __CUDA_ARCH__
    return this->d_adjecencyMatrix[from * this->_size + to];
#else
    return -1;
#endif
}

int GlobalMemoryInstance::hamiltonianCycleWeight(int *cycle) const {
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
