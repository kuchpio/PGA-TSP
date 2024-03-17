#pragma once

#include <cuda_runtime.h>
#include <string>

class Euclidean2D
{
    inline static const std::string CODE = "EUC_2D";
    __device__ __host__ inline static int distance(float x1, float y1, float x2, float y2) {
	    return (int)roundf(sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
    }
};
