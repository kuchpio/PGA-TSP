#include <string>

#pragma once

class IInstance
{
public:
    virtual int size() const = 0;
	virtual int edgeWeight(int from, int to) const = 0;
	virtual int hamiltonianCycleWeight(int *cycle) const = 0;
	virtual ~IInstance() = default;
};

