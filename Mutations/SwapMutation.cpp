#include "SwapMutation.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <curand_kernel.h>

__device__ __host__ void SwapMutation::mutate(int* chromosome, int size, curandState* state) const
{
	int index1 = curand(state) % size;
	int index2 = curand(state) % size;
	std::swap(chromosome[index1], chromosome[index2]);
}

__device__ void mutate(int* chromosome, int size, curandState* state)
{
	int index1 = curand(state) % size;
	int index2 = curand(state) % size;
	std::swap(chromosome[index1], chromosome[index2]);
}

__device__ void intervalMutate(int* chromosome, int size, curandState* state)
{
    int start = curand(state) % size;
    int end = curand(state) % size;
    if (start > end) 
    {
        int temp = start;
        start = end;
        end = temp;
    }

    while (start < end) 
    {
        int temp = chromosome[start];
        chromosome[start] = chromosome[end];
        chromosome[end] = temp;
        start++;
        end--;
    }
}