#include "HostInstance.h"

class HostMemoryInstance: public HostInstance<HostMemoryInstance> {
    private:
        int *_adjecencyMatrix;
        int _size;

    public:
        template<class Metric>
        HostMemoryInstance(const float* x, const float* y, const int size) : _size(size) {
			this->_adjecencyMatrix = new int[size * size];

			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					this->_adjecencyMatrix[i * size + j] = Metric::distance(x[i], y[i], x[j], y[j]);
				}
			}
        }

        int size() const;
        int edgeWeight(const int from, const int to) const;
        int hamiltonianCycleWeight(const int *cycle) const;
        ~HostMemoryInstance();
};
