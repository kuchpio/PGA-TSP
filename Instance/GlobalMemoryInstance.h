#include "IInstance.h"
#include "../Interfaces/IMetric.h"
#include <cuda_runtime.h>

class GlobalMemoryInstance: public IInstance {
    private:
        int *d_adjecencyMatrix;
        int _size;

    public:
        GlobalMemoryInstance(float *x, float *y, IMetric *metric, int size);
        int size() const override;
        __device__ int edgeWeight(int from, int to) const override;
        __device__ int hamiltonianCycleWeight(int *cycle) const override;
        ~GlobalMemoryInstance();
};
