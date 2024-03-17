#include "IInstance.h"
#include "../Interfaces/IMetric.h"

class HostMemoryInstance: public IInstance {
    private:
        int *_adjecencyMatrix;
        int _size;

    public:
        HostMemoryInstance(float *x, float *y, const IMetric *metric, int size);
        __device__ __host__ int size() const override;
        __device__ __host__ int edgeWeight(int from, int to) const override;
        __device__ __host__ int hamiltonianCycleWeight(int *cycle) const override;
        ~HostMemoryInstance();
};
