#pragma once

class ICrossoverOperator
{
public:
	virtual int* crossover(int* parent1, int* parent2, int size, curandState* state) const = 0;
	bool containsVertex(int const* chromosome, int size, int vertex) const
	{
		for (int i = 0; i < size; ++i)
		{
			if (chromosome[i] == vertex)
				return true;
		}
		return false;
	}
	virtual ~ICrossoverOperator() = default;
};
