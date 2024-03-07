#pragma once

#include <curand_kernel.h>
#include "../Interfaces/ISelectionOperator.h"

class RandomSelection : ISelectionOperator
{
	void selection(int* drawnChromosome, int* chromosome, int size, int populationSize, curandState* state) const override;
};
