#pragma once

#include <cuda_runtime.h>

class CeilEuclidean2D
{
public:
    inline static const char* CODE = "CEIL_2D";
    __device__ __host__ inline static int distance(float x1, float y1, float x2, float y2) {
	    return (int)ceilf(sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
    }
};
