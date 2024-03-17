#include <istream>
#include <map>
#include <memory>
#include "../Interfaces/IMetric.h"
#include "../Metrics/Euclidean2D.h"
#include "../Metrics/CeilEuclidean2D.h"
#include "HostMemoryInstance.h"
#include "DeviceMemoryInstanceProxy.h"

class InstanceReader {
    private:
        std::string _name, _type, _comment;
        int _dimension, _selectedMetricIndex;
        float *_x, *_y;                                                                                         
        static constexpr int METRIC_COUNT = 2;
        const IMetric* METRICS[METRIC_COUNT] = {
            new CeilEuclidean2D(), 
            new Euclidean2D(), 
        };
        const size_t METRIC_SIZES[METRIC_COUNT] {
            sizeof(CeilEuclidean2D), 
            sizeof(Euclidean2D)
        };

    public:
        InstanceReader(std::istream& input);
        const HostMemoryInstance* createHostMemoryInstance() const; 

        template<class DeviceMemoryInstance>
        const DeviceMemoryInstanceProxy<DeviceMemoryInstance>* createDeviceMemoryInstance() const {
            return new DeviceMemoryInstanceProxy<DeviceMemoryInstance>(
                this->_x, this->_y, this->_dimension, 
                this->METRICS[this->_selectedMetricIndex], this->METRIC_SIZES[this->_selectedMetricIndex]
            );
        } 

        friend std::ostream& operator<<(std::ostream& output, const InstanceReader& instanceReader);
        ~InstanceReader();
};
