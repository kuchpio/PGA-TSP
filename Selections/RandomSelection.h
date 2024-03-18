#pragma once

#include <curand_kernel.h>
#include "../Interfaces/ISelectionOperator.h"

class RandomSelection : ISelectionOperator
{
	__device__ void selection(int* drawnChromosome, int* chromosome, int size, int populationSize) const override;
};
