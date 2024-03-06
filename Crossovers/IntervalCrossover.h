#pragma once

#include "../Interfaces/ICrossoverOperator.h"

class IntervalCrossover : ICrossoverOperator
{
public:
	int* crossover(int* parent1, int* parent2, int size) const override;
};
