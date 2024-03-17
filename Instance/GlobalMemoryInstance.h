#include "IInstance.h"
#include "../Interfaces/IMetric.h"
#include <cuda_runtime.h>

class GlobalMemoryInstance: public IInstance {
    private:
        int *d_adjecencyMatrix;
        int _size;

    public:
        GlobalMemoryInstance(float *x, float *y, const IMetric *metric, int size);
        __device__ __host__ int size() const override;
        __device__ __host__ int edgeWeight(int from, int to) const override;
        __device__ __host__ int hamiltonianCycleWeight(int *cycle) const override;
        ~GlobalMemoryInstance();
};
