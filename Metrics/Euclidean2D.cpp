#include "Euclidean2D.h"
#include <cuda_runtime.h>
#include <cstdlib>

__device__ __host__ 
float Euclidean2D::distance(float x1, float y1, float x2, float y2) const
{
	return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}
