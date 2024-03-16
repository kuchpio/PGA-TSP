#include "GlobalMemoryInstance.h"
#include <cuda_runtime.h>
#include <iostream>

GlobalMemoryInstance::GlobalMemoryInstance(float *x, float *y, IMetric *metric, int size): _size(size) {
    float *d_x, *d_y;
    cudaError_t status; 

    if ((status = cudaMalloc(&d_x, size * sizeof(float))) != cudaSuccess) {
        std::cerr << "Could not allocate device x coordinate array (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }
    if ((status = cudaMalloc(&d_y, size * sizeof(float))) != cudaSuccess) {
        std::cerr << "Could not allocate device y coordinate array (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }
    if ((status = cudaMalloc(&this->d_adjecencyMatrix, size * size * sizeof(int))) != cudaSuccess) {
        std::cerr << "Could not allocate device adjecency matrix (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }

    if ((status = cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        std::cerr << "Could not copy x coordinate array to device (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }
    if ((status = cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        std::cerr << "Could not copy y coordinate array to device (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }

    // TODO: Run kernel that fills adjecency matrix

    if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
        std::cerr << "Could not synchronize device (" << 
            cudaGetErrorString(status) << ").\n";
        return;
    }

    cudaFree(d_x);
    cudaFree(d_y);
}

int GlobalMemoryInstance::size() const {
    return this->_size;
}

__device__
int GlobalMemoryInstance::edgeWeight(int from, int to) const {
    return this->d_adjecencyMatrix[from * this->_size + to];
}

__device__
int GlobalMemoryInstance::hamiltonianCycleWeight(int *cycle) const {
    int sum = this->edgeWeight(this->_size - 1, 0);

    for (int i = 0; i < this->_size - 1; i++) {
        sum += this->edgeWeight(i, i + 1);
    }

    return sum;
}

GlobalMemoryInstance::~GlobalMemoryInstance() {
    cudaFree(this->d_adjecencyMatrix);
}
