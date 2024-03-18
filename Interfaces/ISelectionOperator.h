#pragma once

#include <curand_kernel.h>

class ISelectionOperator
{
public:
	virtual void selection(int* drawnChromosome, int* chromosome, int size, int populationSize) const = 0;
	float calculateFitness(int* chromosome, int size) const {
		return 0;
	} // add all edges from tex2D i->j size + 1 edges
	virtual ~ISelectionOperator() = default;
};
