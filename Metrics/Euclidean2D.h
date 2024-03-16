#pragma once

#include "../Interfaces/IMetric.h"
#include <string>
#include <cuda_runtime.h>

class Euclidean2D : public IMetric
{
private:
    const std::string CODE = "EUC_2D";
public:
	__device__ __host__ int distance(float x1, float y1, float x2, float y2) const override;
    const std::string& code() const override;
};
