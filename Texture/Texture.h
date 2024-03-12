#pragma once

template <typename T>
class TextureClass
{
private:
	cudaArray* cuArray;
	cudaTextureObject_t texObj;

public:
	TextureClass();
	void InitializeTexture(T** data, size_t width, size_t height);
	~TextureClass();
};