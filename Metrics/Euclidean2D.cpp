#include "Euclidean2D.h"
#include <cmath>
#include <cuda_runtime.h>

__device__ __host__ 
int Euclidean2D::distance(float x1, float y1, float x2, float y2) const
{
	return (int)roundf(sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

const std::string& Euclidean2D::code() const {
    return this->CODE;
}

