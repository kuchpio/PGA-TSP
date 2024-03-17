#include "IInstance.h"
#include "../Interfaces/IMetric.h"
#include <iostream>

template<class DeviceMemoryInstance>
__global__ 
void sizeKernel(const DeviceMemoryInstance* deviceMemoryInstance, int *size) {
    *size = deviceMemoryInstance->size();
}

template<class DeviceMemoryInstance>
__global__ 
void edgeWeightKernel(const DeviceMemoryInstance* deviceMemoryInstance, const int from, const int to, int *edgeWeight) {
    *edgeWeight = deviceMemoryInstance->edgeWeight(from, to);
}

template<class DeviceMemoryInstance>
__global__ 
void hamiltonianCycleWeightKernel(const DeviceMemoryInstance* deviceMemoryInstance, const int *cycle, int *cycleWeight) {
    *cycleWeight = deviceMemoryInstance->hamiltonianCycleWeight(cycle);
}

template<class DeviceMemoryInstance>
class DeviceMemoryInstanceProxy: public IInstance {
private:
    DeviceMemoryInstance* d_deviceMemoryInstance;

public:
    DeviceMemoryInstanceProxy(const float *x, const float *y, const int vertexCount, const IMetric *metric, const size_t metricSize) {
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
        const DeviceMemoryInstance *h_deviceMemoryInstance = new DeviceMemoryInstance(
            (const float*) d_x, (const float*) d_y, vertexCount, (const IMetric*) d_metric
        );

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
        int *d_size, h_size; 

        if ((status = cudaMalloc(&d_size, sizeof(int))) != cudaSuccess) {
            std::cerr << "Could not allocate device size output variable (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        sizeKernel<DeviceMemoryInstance><<<1, 1>>>(this->d_deviceMemoryInstance, d_size);

        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        if ((status = cudaMemcpy(&h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
            std::cerr << "Could not copy device memory size to device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        cudaFree(d_size);

        return h_size;
#endif
    }

    __device__ __host__
    int edgeWeight(const int from, const int to) const override {
#ifdef __CUDA_ARCH__
        return -1;
#else
        cudaError_t status; 
        int *d_weight, h_weight; 

        if ((status = cudaMalloc(&d_weight, sizeof(int))) != cudaSuccess) {
            std::cerr << "Could not allocate device weight output variable (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        edgeWeightKernel<DeviceMemoryInstance><<<1, 1>>>(this->d_deviceMemoryInstance, from, to, d_weight);

        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        if ((status = cudaMemcpy(&h_weight, d_weight, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
            std::cerr << "Could not copy device memory weight to device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        cudaFree(d_weight);

        return h_weight;
#endif
    }

    __device__ __host__
    int hamiltonianCycleWeight(const int *cycle) const override {
#ifdef __CUDA_ARCH__
        return -1;
#else
        cudaError_t status; 
        int *d_cycle, *d_weight, h_weight; 

        if ((status = cudaMalloc(&d_weight, sizeof(int))) != cudaSuccess) {
            std::cerr << "Could not allocate device weight output variable (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        if ((status = cudaMalloc(&d_cycle, this->size() * sizeof(int))) != cudaSuccess) {
            std::cerr << "Could not allocate device cycle variable (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        hamiltonianCycleWeightKernel<DeviceMemoryInstance><<<1, 1>>>(this->d_deviceMemoryInstance, (const int*) d_cycle, d_weight);

        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        if ((status = cudaMemcpy(&h_weight, d_weight, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) {
            std::cerr << "Could not copy device memory weight to device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        cudaFree(d_weight);
        cudaFree(d_cycle);

        return h_weight;
#endif
    }

    const DeviceMemoryInstance* deviceMemoryInstance() const {
        return this->d_deviceMemoryInstance;
    }

    ~DeviceMemoryInstanceProxy() {
        cudaFree(this->d_deviceMemoryInstance);
    }
};
