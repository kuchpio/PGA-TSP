#pragma once

class ICrossoverOperator
{
public:
	virtual int* crossover(int* parent1, int* parent2, int size, curandState* state) const = 0;
	virtual ~ICrossoverOperator() = default;
};
