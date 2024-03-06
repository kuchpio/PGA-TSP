#pragma once

#include "../Interfaces/ISelectionOperator.h"

class RandomSelection : ISelectionOperator
{ 
	void selection(int** population, int* chromosome, int size, int populationSize) const override;
};
