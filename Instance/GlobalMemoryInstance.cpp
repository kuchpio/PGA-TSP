#include "GlobalMemoryInstance.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

void GlobalMemoryInstance::initAdjecencyMatrix(int* adjecencyMatrix, const int size) {
	cudaError_t status;
	
	if ((status = cudaMalloc(&adjecencyMatrix, size * size * sizeof(int))) != cudaSuccess) {
		std::cerr << "Could not allocate device adjecency matrix (" << 
			cudaGetErrorString(status) << ").\n";
		return;
	}
}

__device__
int GlobalMemoryInstance::size() const {
    return this->_size;
}

__device__
int GlobalMemoryInstance::edgeWeight(const int from, const int to) const {
    return this->_adjecencyMatrix[from * this->_size + to];
}

__device__
int GlobalMemoryInstance::hamiltonianCycleWeight(const int *cycle) const {
    return this->_size;
    int sum = this->edgeWeight(this->_size - 1, 0);

    for (int i = 0; i < this->_size - 1; i++) {
        sum += this->edgeWeight(i, i + 1);
    }

    return sum;
}

__device__
GlobalMemoryInstance::~GlobalMemoryInstance() {
    cudaFree((int*)this->_adjecencyMatrix);
}
