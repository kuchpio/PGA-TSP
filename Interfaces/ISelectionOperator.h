#pragma once

#include <curand_kernel.h>

class ISelectionOperator
{
public:
	virtual void selection(int* drawnChromosome, int* chromosome, int size, int populationSize) const = 0;
	virtual ~ISelectionOperator() = default;
};

float calculateFitness(int* chromosome, int size, float** distance_matrix);