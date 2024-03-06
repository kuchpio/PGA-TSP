#pragma once

#include "../Interfaces/IMutationOperator.h"

class SwapMutation : IMutationOperator
{
public:
	void mutate(int* chromosome, int size) const override;
};
