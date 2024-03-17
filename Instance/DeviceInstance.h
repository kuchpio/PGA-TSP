#pragma once
#include <cuda_runtime.h>

template<class Implementation>
class DeviceInstance {
public:
    __device__ int size() const {
        return static_cast<Implementation*>(this)->size();
    }
    __device__ int edgeWeight(const int from, const int to) const {
        return static_cast<Implementation*>(this)->edgeWeight(from, to);
    }
    __device__ int hamiltonianCycleWeight(const int* cycle) const {
        return static_cast<Implementation*>(this)->edgeWeight(cycle);
    }
};
