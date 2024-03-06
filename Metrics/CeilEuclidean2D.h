#pragma once

#include "../Interfaces/IMetric.h"

class CeilEuclidean2D : IMetric<int>
{
public:
	int distance(int x1, int y1, int x2, int y2) const override;
};
