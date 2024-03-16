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
    return this->_size;
}

int HostMemoryInstance::edgeWeight(int from, int to) const {
    return this->_adjecencyMatrix[from * this->_size + to];
}

int HostMemoryInstance::hamiltonianCycleWeight(int *cycle) const {
    int sum = this->edgeWeight(this->_size - 1, 0);

    for (int i = 0; i < this->_size - 1; i++) {
        sum += this->edgeWeight(i, i + 1);
    }

    return sum;
}

HostMemoryInstance::~HostMemoryInstance() {
    delete[] this->_adjecencyMatrix;
}
