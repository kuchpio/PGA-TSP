#include "IInstance.h"
#include "../Interfaces/IMetric.h"

class HostMemoryInstance: public IInstance {
    private:
        int *_adjecencyMatrix;
        int _size;

    public:
        HostMemoryInstance(float *x, float *y, const IMetric *metric, int size);
        int size() const override;
        int edgeWeight(int from, int to) const override;
        int hamiltonianCycleWeight(int *cycle) const override;
        ~HostMemoryInstance();
};
