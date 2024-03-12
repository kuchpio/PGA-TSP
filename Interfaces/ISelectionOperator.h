#pragma once

#include <curand_kernel.h>

class ISelectionOperator
{
public:
	virtual void selection(int* drawnChromosome, int* chromosome, int size, int populationSize, curandState* state) const = 0;
	float calculateFitness(int* chromosome, int size) const;
	virtual ~ISelectionOperator() = default;
};
