#pragma once

class ISelectionOperator
{
public:
	virtual void selection(int** population, int* chromosome, int size, int populationSize) const = 0; // int** global memory :/
	float calculateFitness(int* chromosome, int size) const {
		return 0;
	} // add all edges from tex2D i->j size + 1 edges
	virtual ~ISelectionOperator() = default;
};
