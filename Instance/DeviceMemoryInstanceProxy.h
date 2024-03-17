#include <iostream>
#include <cuda_runtime.h>
#include "HostInstance.h"

template<class DeviceMemoryInstance>
__global__ 
void newKernel(DeviceMemoryInstance* deviceMemoryInstance, const int *adjecencyMatrix, int size) {
	*deviceMemoryInstance = new DeviceMemoryInstance(adjecencyMatrix, size);
}

template<class DeviceMemoryInstance>
__global__ 
void deleteKernel(DeviceMemoryInstance* deviceMemoryInstance) {
    delete deviceMemoryInstance;
}

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
class DeviceMemoryInstanceProxy: public HostInstance<DeviceMemoryInstanceProxy> {
private:
    DeviceMemoryInstance* d_deviceMemoryInstance;

public:
    template<class Metric>
    DeviceMemoryInstanceProxy(const float *x, const float *y, const int size) {
        // Copy x, y to device
        float *d_x, *d_y;
        int* d_adjecencyMatrix;
        cudaError_t status; 

        if ((status = cudaMalloc(&d_x, size * sizeof(float))) != cudaSuccess) {
            std::cerr << "Could not allocate device x coordinate array (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMalloc(&d_y, size * sizeof(float))) != cudaSuccess) {
            std::cerr << "Could not allocate device y coordinate array (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy x coordinate array to device (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }
        if ((status = cudaMemcpy(d_y, y, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy y coordinate array to device (" << 
                cudaGetErrorString(status) << ").\n";
            return;
        }

        DeviceMemoryInstance::initializeAdjecencyMatrix(d_adjecencyMatrix, size);
        DeviceMemoryInstance::fillAdjecencyMatrix<Metric>(d_adjecencyMatrix, d_x, d_y, size);
        newKernel<DeviceMemoryInstance><<<1, 1>>>(this->d_deviceMemoryInstance, d_adjecencyMatrix, size);

        if ((status = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cerr << "Could not synchronize device (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        cudaFree(d_x);
        cudaFree(d_y);
    }

    int size() const {
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
            std::cerr << "Could not copy device memory size to host (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        cudaFree(d_size);

        return h_size;
    }

    int edgeWeight(const int from, const int to) const override {
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
            std::cerr << "Could not copy device memory weight to host (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        cudaFree(d_weight);

        return h_weight;
    }

    int hamiltonianCycleWeight(const int *cycle) const override {
        cudaError_t status; 
        int *d_cycle, *d_weight, h_weight; 
        int size = this->size();

        if ((status = cudaMalloc(&d_weight, sizeof(int))) != cudaSuccess) {
            std::cerr << "Could not allocate device weight output variable (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        if ((status = cudaMalloc(&d_cycle, size * sizeof(int))) != cudaSuccess) {
            std::cerr << "Could not allocate device cycle variable (" << 
                cudaGetErrorString(status) << ").\n";
            return -1;
        }

        if ((status = cudaMemcpy(d_cycle, cycle, size * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cerr << "Could not copy device memory weight to device (" << 
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
    }

    const DeviceMemoryInstance* deviceMemoryInstance() const {
        return this->d_deviceMemoryInstance;
    }

    ~DeviceMemoryInstanceProxy() {
        deleteKernel<DeviceMemoryInstance><<<1, 1>>>(this->d_deviceMemoryInstance);
    }
};
