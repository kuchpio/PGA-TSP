#ifndef __GLOBAL_MEMORY_INSTANCE_H__
#define __GLOBAL_MEMORY_INSTANCE_H__

// https://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-functions

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>

namespace tsp {

	typedef struct TextureMemoryInstance {
		const cudaTextureObject_t textureObject = 0;
		const cudaArray_t array = NULL;
		const int size = 0;
	} TextureMemoryInstance;

	template<class Metric>
	__global__
	void fillAdjecencyMatrixKernel(int* adjecencyMatrix, const float* x, const float* y, const int size, Metric metric) {
		int tid, row, col;
		for (tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size * size; tid += blockDim.x * gridDim.x) {
			row = tid / size;
			col = tid % size;
			adjecencyMatrix[tid] = distance(metric, x[row], y[row], x[col], y[col]);
		}
	}

	template <typename Metric>
	const TextureMemoryInstance initInstance(const float* x, const float* y, const int size, Metric metric) {
		float* d_x, * d_y;
		int* d_adjecencyMatrix;
		cudaTextureObject_t textureObject = 0;
		cudaArray_t array = NULL;

		if (cudaMalloc(&d_x, size * sizeof(float)) != cudaSuccess)
			return { };
		if (cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return { };
		if (cudaMalloc(&d_y, size * sizeof(float)) != cudaSuccess)
			return { };
		if (cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return { };
		if (cudaMalloc(&d_adjecencyMatrix, size * size * sizeof(int)) != cudaSuccess)
			return { };

		fillAdjecencyMatrixKernel << <4, 256 >> > (d_adjecencyMatrix, d_x, d_y, size, metric);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return { };
		
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
		if (cudaMallocArray(&array, &channelDesc, size, size) != cudaSuccess)
			return { };
		if (cudaMemcpy2DToArray(array, 0, 0, d_adjecencyMatrix, size * sizeof(int), size * sizeof(int), size, cudaMemcpyDeviceToDevice) != cudaSuccess)
			return { };

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		if (cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr) != cudaSuccess)
			return { };

		cudaFree(d_adjecencyMatrix);
		cudaFree(d_x);
		cudaFree(d_y);

		return { textureObject, array, size };
	}

	bool isValid(TextureMemoryInstance instance) {
		return instance.textureObject != 0;
	}

	void destroyInstance(TextureMemoryInstance instance) {
		cudaFreeArray(instance.array);
		cudaDestroyTextureObject(instance.textureObject);
	}

	__device__
	int size(const TextureMemoryInstance instance) {
		return instance.size;
	}

	__device__
	int edgeWeight(const TextureMemoryInstance instance, const int from, const int to) {
		return tex2D<int>(instance.textureObject, from, to);
	}

	__device__
	int hamiltonianCycleWeight(const TextureMemoryInstance instance, const int* cycle) {
		int sum = edgeWeight(instance, size(instance) - 1, 0);

		for (int i = 0; i < size(instance) - 1; i++) {
			sum += edgeWeight(instance, i, i + 1);
		}

		return sum;
	}

}

#endif
