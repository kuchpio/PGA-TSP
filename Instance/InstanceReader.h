#include <istream>
#include <map>
#include <memory>
#include "../Interfaces/IMetric.h"
#include "../Metrics/Euclidean2D.h"
#include "../Metrics/CeilEuclidean2D.h"
#include "HostMemoryInstance.h"

class InstanceReader {
    private:
        std::string _name, _type, _comment;
        int _dimension;
        const IMetric *_metric;
        static constexpr int METRIC_COUNT = 2;
        const IMetric* METRICS[METRIC_COUNT] = {
            new CeilEuclidean2D(), 
            new Euclidean2D(), 
        };
        float *_x, *_y;                                                                                         

    public:
        InstanceReader(std::istream& input);
        const HostMemoryInstance* createHostMemoryInstance() const; 
        friend std::ostream& operator<<(std::ostream& output, const InstanceReader& instanceReader);
        ~InstanceReader();
};
