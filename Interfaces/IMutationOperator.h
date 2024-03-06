#pragma once

class IMutationOperator
{
public:
	virtual void mutate(int* chromosome, int size) const = 0;
	virtual ~IMutationOperator() = default;
};
