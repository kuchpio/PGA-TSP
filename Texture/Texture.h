#pragma once

template <typename T>
class TextureClass
{
public:
	cudaTextureObject_t tex = 0;
	cudaArray* cuArray;

	TextureClass();
	void InitializeTexture(T** data, size_t width, size_t height);
	~TextureClass();
};
