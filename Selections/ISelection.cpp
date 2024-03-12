#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../Interfaces/ISelectionOperator.h"

float ISelectionOperator::calculateFitness(int* chromosome, int size) const
{
	float sum = 1/*tex2D<float>(tex, chromosome[0], chromosome[size - 1])*/;

	for (int i = 0; i < size - 1; ++i)
	{
		sum += 1 /*tex2D<float>(tex, chromosome[i], chromosome[i + 1])*/;
	}
	return sum;
}