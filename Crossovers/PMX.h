#ifndef __PMX_CROSSOVER_H__
#define __PMX_CROSSOVER_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tsp {
	// Partially Mapped Crossover (PMX)
	__device__
		void PMX(int* a, int* b, const int size, curandState* state) {
		int left = curand(state) % size;
		int right = curand(state) % size;

		if (left > right) {
			int tmp = left;
			left = right;
			right = tmp;
		}
		int* c = new int[size];
		int* d = new int[size];
		for (int i = 0; i < size; i++)
		{
			c[i] = a[i];
			d[i] = b[i];
		}
		int tmp;
		for (int i = left; i <= right; i++)
		{
			bool done0 = false;
			bool done1 = false;
			for (int j = 0; j < size; j++)
			{
				done0 = false;
				done1 = false;
				if (c[j] == b[i])
				{
					done0 = true;
					c[j] = c[i];
				}
				if (d[j] == a[i])
				{
					done1 = true;
					d[j] = d[i];
				}
				if (done0 && done1) break;
			}
		}
		for (int i = 0; i < size; i++)
		{
			if ((i >= left) && (i <= right)) {
				tmp = a[i];
				a[i] = b[i];
				b[i] = tmp;
			}
			else {
				a[i] = c[i];
				b[i] = d[i];
			}
		}
		delete[] c;
		delete[] d;
	}
}

#endif
