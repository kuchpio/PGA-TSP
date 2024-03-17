#include <string>
#include <cuda_runtime.h>

#pragma once

class IInstance
{
public:
    __device__ __host__ virtual int size() const = 0;
	__device__ __host__ virtual int edgeWeight(int from, int to) const = 0;
	__device__ __host__ virtual int hamiltonianCycleWeight(int *cycle) const = 0;
	virtual ~IInstance() = default;
};

