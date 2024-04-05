#include "ISelectionOperator.h"

float calculateFitness(int* chromosome, int size, float** distance_matrix)
{
	float sum = distance_matrix[chromosome[size - 1]][chromosome[0]];
	for (int i = 0; i < size - 1; ++i)
	{
		sum += distance_matrix[chromosome[i]][chromosome[i + 1]];
	}
	return sum;
}