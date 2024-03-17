#include "IInstance.h"
#include "../Interfaces/IMetric.h"
#include <iostream>

template<class DeviceMemoryInstance>
class DeviceMemoryInstanceProxy: public IInstance {
private:
    DeviceMemoryInstance* d_deviceMemoryInstance;

public:
    DeviceMemoryInstanceProxy(float *x, float *y, int vertexCount, const IMetric *metric, size_t metricSize) {
        // Copy x, y, metric to device
        float *d_x, *d_y;
        IMetric *d_metric;
        cudaError_t status; 

        if ((status = cudaMalloc(&d_x, vertexCount * sizeof(float))) != cudaSuccess) {
            std::cerr << "Could not allocate device x coordinate array (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMalloc(&d_y, vertexCount * sizeof(float))) != cudaSuccess) {
            std::cerr << "Could not allocate device y coordinate array (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMalloc(&d_metric, metricSize)) != cudaSuccess) {
            std::cerr << "Could not allocate device y coordinate array (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }

        if ((status = cudaMemcpy(d_x, x, vertexCount * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy x coordinate array to device (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMemcpy(d_y, y, vertexCount * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy y coordinate array to device (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMemcpy(d_metric, metric, metricSize, cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy metric to device (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }

        // Initialize deviceMemoryInstance on host
        const DeviceMemoryInstance *h_deviceMemoryInstance = new DeviceMemoryInstance(d_x, d_y, (const IMetric*) d_metric, vertexCount);

        // Memcpy deviceMemoryInstance to device
        if ((status = cudaMalloc(&this->d_deviceMemoryInstance, sizeof(DeviceMemoryInstance))) != cudaSuccess) {
            std::cerr << "Could not allocate device memory instance (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMemcpy(this->d_deviceMemoryInstance, h_deviceMemoryInstance, sizeof(DeviceMemoryInstance), cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy device memory instance to device (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }

        // Deallocate deviceMemoryInstance on host
        delete h_deviceMemoryInstance;
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_metric);
    }

    __device__ __host__
    int size() const override {
#ifdef __CUDA_ARCH__
        return -1;
#else
        cudaError_t status; 

        // Memcpy deviceMemoryInstance->size() to host
        
        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return 0;
        }

        return 0;
#endif
    }

    __device__ __host__
    int edgeWeight(int from, int to) const override {
#ifdef __CUDA_ARCH__
        return -1;
#else
        cudaError_t status; 

        // Memcpy deviceMemoryInstance->edgeWeight(int from, int to) to host
        
        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return 0;
        }

        return 0;
#endif
    }

    __device__ __host__
    int hamiltonianCycleWeight(int *cycle) const override {
#ifdef __CUDA_ARCH__
        return -1;
#else
        cudaError_t status; 

        // Memcpy deviceMemoryInstance->hamiltonianCycleWeight(int *cycle) to host
        
        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return 0;
        }

        return 0;
#endif
    }

    const DeviceMemoryInstance* deviceMemoryInstance() const {
        return this->d_deviceMemoryInstance;
    }

    ~DeviceMemoryInstanceProxy() {
        cudaFree(this->d_deviceMemoryInstance);
    }
};
