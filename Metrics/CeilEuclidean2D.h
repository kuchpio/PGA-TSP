#pragma once

#include "../Interfaces/IMetric.h"
#include <string>
#include <cuda_runtime.h>

class CeilEuclidean2D : public IMetric
{
private:
    const std::string CODE = "CEIL_2D";
public:
	__device__ __host__ int distance(float x1, float y1, float x2, float y2) const override;
    const std::string& code() const override;
};
