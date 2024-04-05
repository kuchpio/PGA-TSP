// https://on-demand.gputechconf.com/gtc-express/2011/presentations/texture_webinar_aug_2011.pdf
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-functions
#include "Texture.h"
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cstdlib>

template <typename T>
TextureClass<T>::TextureClass() : cuArray(nullptr)
{
}

template <typename T>
void TextureClass<T>::InitializeTexture(T** data, size_t width, size_t height)
{
	// Define CUDA array format
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

	// Allocate CUDA array in device memory
	cudaMallocArray(&cuArray, &channelDesc, width, height);

	// Copy to CUDA array
	cudaMemcpy2DToArray(cuArray, 0, 0, data, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Create texture object
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
}

template<typename T>
TextureClass<T>::~TextureClass()
{
	if (cuArray != nullptr)
	{
		cudaFree(cuArray);
	}

	if (tex != 0)
	{
		cudaDestroyTextureObject(tex);
	}
}