#include "IInstance.h"
#include "../Interfaces/IMetric.h"
#include <cuda_runtime.h>

class GlobalMemoryInstance: public IInstance {
    private:
        int *d_adjecencyMatrix;
        int _size;

    public:
        GlobalMemoryInstance(const float *x, const float *y, const int size, const IMetric *metric);
        __device__ __host__ int size() const override;
        __device__ __host__ int edgeWeight(const int from, const int to) const override;
        __device__ __host__ int hamiltonianCycleWeight(const int *cycle) const override;
        ~GlobalMemoryInstance();
};
