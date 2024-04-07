#ifndef __TEXTURE_MEMORY_INSTANCE_H__
#define __TEXTURE_MEMORY_INSTANCE_H__

// https://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-functions

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>

namespace tsp {

	typedef struct TextureMemoryInstance {
		cudaTextureObject_t textureObject = 0;
		cudaArray_t array = NULL;
		int size = 0;
	} TextureMemoryInstance;

	template <typename Metric>
	bool initInstance(TextureMemoryInstance *instance, const float* x, const float* y, const int size, Metric metric) {
		float* d_x, * d_y;
		int* d_adjecencyMatrix;
		cudaTextureObject_t textureObject = 0;
		cudaArray_t array = NULL;

		if (cudaMalloc(&d_x, size * sizeof(float)) != cudaSuccess)
			return false;
		if (cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return false;
		if (cudaMalloc(&d_y, size * sizeof(float)) != cudaSuccess)
			return false;
		if (cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
			return false;
		if (cudaMalloc(&d_adjecencyMatrix, size * size * sizeof(int)) != cudaSuccess)
			return false;

		fillAdjecencyMatrixKernel << <4, 256 >> > (d_adjecencyMatrix, d_x, d_y, size, metric);

		if (cudaDeviceSynchronize() != cudaSuccess)
			return false;
		
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
		if (cudaMallocArray(&array, &channelDesc, size, size) != cudaSuccess)
			return false;
		if (cudaMemcpy2DToArray(array, 0, 0, d_adjecencyMatrix, size * sizeof(int), size * sizeof(int), size, cudaMemcpyDeviceToDevice) != cudaSuccess)
			return false;

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
			return false;

		cudaFree(d_adjecencyMatrix);
		cudaFree(d_x);
		cudaFree(d_y);

		instance->textureObject = textureObject;
		instance->array = array;
		instance->size = size;

		return true;
	}

	void destroyInstance(TextureMemoryInstance instance) {
		cudaFreeArray(instance.array);
		cudaDestroyTextureObject(instance.textureObject);
	}

	__device__ __host__
	int size(const TextureMemoryInstance instance) {
		return instance.size;
	}

	__device__
	int edgeWeight(const TextureMemoryInstance instance, const int from, const int to) {
		return tex2D<int>(instance.textureObject, from, to);
	}

	__device__
	int hamiltonianCycleWeight(const TextureMemoryInstance instance, const int* cycle) {
		int sum = edgeWeight(instance, cycle[size(instance) - 1], cycle[0]);

		for (int i = 0; i < size(instance) - 1; i++) {
			sum += edgeWeight(instance, cycle[i], cycle[i + 1]);
		}

		return sum;
	}

}

#endif
