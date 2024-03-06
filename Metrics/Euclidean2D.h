#pragma once

#include "../Interfaces/IMetric.h"

class Euclidean2D : IMetric<float>
{
public:
	float distance(float x1, float y1, float x2, float y2) const override;
};
