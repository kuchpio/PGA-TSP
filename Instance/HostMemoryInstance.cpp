#include "HostMemoryInstance.h"

HostMemoryInstance::HostMemoryInstance(float *x, float *y, const IMetric *metric, int size): _size(size) {
    this->_adjecencyMatrix = new int[size * size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            this->_adjecencyMatrix[i * size + j] = 
                metric->distance(x[i], y[i], x[j], y[j]);
        }
    }
}

int HostMemoryInstance::size() const {
#ifdef __CUDA_ARCH__
    return -1;
#else
    return this->_size;
#endif
}

int HostMemoryInstance::edgeWeight(int from, int to) const {
#ifdef __CUDA_ARCH__
    return -1;
#else
    return this->_adjecencyMatrix[from * this->_size + to];
#endif
}

int HostMemoryInstance::hamiltonianCycleWeight(int *cycle) const {
#ifdef __CUDA_ARCH__
    return -1;
#else
    int sum = this->edgeWeight(this->_size - 1, 0);

    for (int i = 0; i < this->_size - 1; i++) {
        sum += this->edgeWeight(i, i + 1);
    }

    return sum;
#endif
}

HostMemoryInstance::~HostMemoryInstance() {
    delete[] this->_adjecencyMatrix;
}
