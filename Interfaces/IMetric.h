#include <string>
#include <cuda_runtime.h>

#pragma once

class IMetric
{
public:
	__device__ __host__ virtual int distance(float x1, float y1, float x2, float y2) const = 0;
    virtual const std::string& code() const = 0;
	virtual ~IMetric() = default;
};
