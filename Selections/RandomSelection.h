#pragma once

#include <curand_kernel.h>
#include "../Interfaces/ISelectionOperator.h"

class RandomSelection : ISelectionOperator
{
	void selection(int** population, int* chromosome, int size, int populationSize, curandState* state) const override;
};
