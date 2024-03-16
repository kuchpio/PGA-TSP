#include "CeilEuclidean2D.h"
#include <cmath>
#include <cuda_runtime.h>

__device__ __host__ 
int CeilEuclidean2D::distance(float x1, float y1, float x2, float y2) const
{
	return (int)ceilf(sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

const std::string& CeilEuclidean2D::code() const {
    return this->CODE;
}
