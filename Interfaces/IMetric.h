#pragma once

template<typename T>
class IMetric
{
public:
	virtual T distance(T x1, T y1, T x2, T y2) const = 0;
	virtual ~IMetric() = default;
};
