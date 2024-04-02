#pragma once

#include <curand_kernel.h>
#include "../Interfaces/IMutationOperator.h"

class SwapMutation : IMutationOperator
{
public:
	void mutate(int* chromosome, int size, curandState* state) const override;
};

void mutate(int* chromosome, int size, curandState* state);

void intervalMutate(int* chromosome, int size, curandState* state);
