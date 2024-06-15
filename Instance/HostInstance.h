#ifndef __HOST_INSTANCE_H__
#define __HOST_INSTANCE_H__

#include <cuda_runtime.h>
#include <iostream>

namespace tsp {

    class IHostInstance {
    public:
        virtual int size() const = 0;
        virtual int edgeWeight(const int from, const int to) const = 0;
    };

    class HostMemoryInstance : public IHostInstance {
    private:
        int* _adjecencyMatrix;
        int _size;

    public:

        template<typename Metric>
        HostMemoryInstance(const float* x, const float* y, const int size, Metric metric) : _size(size) {
            this->_adjecencyMatrix = new int[size * size];

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    this->_adjecencyMatrix[i * size + j] = distance(metric, x[i], y[i], x[j], y[j]);
                }
            }
        }

        int size() const override {
            return this->_size;
        }

        int edgeWeight(const int from, const int to) const override {
            return this->_adjecencyMatrix[from * this->_size + to];
        }

        ~HostMemoryInstance() {
            delete[] this->_adjecencyMatrix;
        }
    };

	template<class DeviceInstance>
	__global__
		void sizeKernel(const DeviceInstance deviceInstance, int* out) {
		*out = size(deviceInstance);
	}

	template<class DeviceInstance>
	__global__
		void edgeWeightKernel(const DeviceInstance deviceInstance, const int from, const int to, int* out) {
		*out = edgeWeight(deviceInstance, from, to);
	}

    template<typename DeviceInstance>
    class DeviceInstanceHostAdapter : public IHostInstance {
    private:
        const DeviceInstance _deviceInstance;

    public:

        template<typename Metric>
        DeviceInstanceHostAdapter(const float* x, const float* y, const int size, Metric metric)
            : _deviceInstance([x, y, size, metric]() {
					DeviceInstance deviceInstance;

					if (!initInstance(&deviceInstance, x, y, size, metric)) {
						std::cerr << "Could not initialize device instance.\n";
					}

					return deviceInstance;
                }()) { }

        int size() const override {
            cudaError_t status;
            int* d_size, h_size;

            if ((status = cudaMalloc(&d_size, sizeof(int))) != cudaSuccess) {
                std::cerr << "Could not allocate device size output variable (" <<
                    cudaGetErrorString(status) << ").\n";
                return -1;
            }

            sizeKernel << <1, 1 >> > (this->_deviceInstance, d_size);

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
            int* d_weight, h_weight;

            if ((status = cudaMalloc(&d_weight, sizeof(int))) != cudaSuccess) {
                std::cerr << "Could not allocate device weight output variable (" <<
                    cudaGetErrorString(status) << ").\n";
                return -1;
            }

            edgeWeightKernel << <1, 1 >> > (this->_deviceInstance, from, to, d_weight);

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

        const DeviceInstance deviceInstance() const {
            return this->_deviceInstance;
        }
    };

}

#endif
