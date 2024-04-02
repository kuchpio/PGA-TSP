#pragma once

#include <curand_kernel.h>
#include "../Interfaces/ISelectionOperator.h"

class RandomSelection : ISelectionOperator
{
	void selection(int* drawnChromosome, int* chromosome, int size, int populationSize) const override;
};

void selection(int* drawnChromosome, int* chromosome, int size, float fitness1, float fitness2);

int rouletteWheelSelection(float* fitness, int populationSize, curandState* state, float total_fitness);
