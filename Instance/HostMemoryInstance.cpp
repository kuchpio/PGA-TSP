#include "HostMemoryInstance.h"

int HostMemoryInstance::size() const {
    return this->_size;
}

int HostMemoryInstance::edgeWeight(const int from, const int to) const {
    return this->_adjecencyMatrix[from * this->_size + to];
}

int HostMemoryInstance::hamiltonianCycleWeight(const int *cycle) const {
    int sum = this->edgeWeight(this->_size - 1, 0);

    for (int i = 0; i < this->_size - 1; i++) {
        sum += this->edgeWeight(i, i + 1);
    }

    return sum;
}

HostMemoryInstance::~HostMemoryInstance() {
    delete[] this->_adjecencyMatrix;
}
