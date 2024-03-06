#include "CeilEuclidean2D.h"
#include <cuda_runtime.h>
#include <cstdlib>

__device__ __host__ 
int CeilEuclidean2D::distance(int x1, int y1, int x2, int y2) const
{
	return (int)ceilf(sqrtf((float)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))));
}
