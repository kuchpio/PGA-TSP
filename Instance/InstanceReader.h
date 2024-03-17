#include <istream>
#include <map>
#include <memory>
#include "HostInstance.h"
#include "../Metrics/Euclidean2D.h"

class InstanceReader {
    private:
        std::string _name, _type, _comment, _metric;
        int _dimension;
        float *_x, *_y; 

    public:
        InstanceReader(std::istream& input);

        template<class Implementation>
        const HostInstance<Implementation>* createInstance() const {
            return new Implementation::Implementation<Euclidean2D>(this->_x, this->_y, this->_dimension);
        } 

        friend std::ostream& operator<<(std::ostream& output, const InstanceReader& instanceReader);
        ~InstanceReader();
};
