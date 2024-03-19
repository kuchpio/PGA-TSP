#pragma once

#include <curand_kernel.h>
#include "../Interfaces/ICrossoverOperator.h"

class IntervalCrossover : ICrossoverOperator
{
public:
	int* crossover(int* parent1, int* parent2, int size, curandState* state) const override;
};

bool containsVertex(int const* chromosome, int size, int vertex);
int* crossover(int* parent1, int* parent2, int size, curandState* state);