#pragma once

#include <curand_kernel.h>
#include "../Interfaces/ICrossoverOperator.h"

class IntervalCrossover : ICrossoverOperator
{
public:
	int* crossover(int* parent1, int* parent2, int size, curandState* state) const override;
};


int* crossover(int* parent1, int* parent2, int size, curandState* state);