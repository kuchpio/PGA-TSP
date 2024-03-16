#include "InstanceReader.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <istream>
#include <map>
#include <sstream>
#include <string>

void trim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
        return !std::isspace(c);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {
        return !std::isspace(c);
    }).base(), s.end());
}

InstanceReader::InstanceReader(std::istream& input) {
    std::stringstream lineStream;
    std::string line, key, value;
    int node;
    float x, y;
    std::map<std::string, std::string> keyValuePairs;

    // Read specification
    while (std::getline(input, line)) {
        if (line == "NODE_COORD_SECTION")
            break;

        std::stringstream lineStream(line);

        if (std::getline(lineStream, key, ':') && std::getline(lineStream, value)) {
            trim(key);
            trim(value);
            keyValuePairs[key] = value;
        }
    }

    // Process specification
    this->_type = keyValuePairs["TYPE"];
    this->_name = keyValuePairs["NAME"];
    this->_comment = keyValuePairs["COMMENT"];
    this->_dimension = std::stoi(keyValuePairs["DIMENSION"]);
    std::string metricCode = keyValuePairs["EDGE_WEIGHT_TYPE"]; 
    this->_metric = *std::find_if(this->METRICS, this->METRICS + this->METRIC_COUNT, 
        [metricCode](const IMetric *m) {
            return m->code() == metricCode;
        });
    this->_x = new float[this->_dimension];
    this->_y = new float[this->_dimension];

    // Read data
    while (input >> node >> x >> y) {
        this->_x[node - 1] = x;
        this->_y[node - 1] = y;
    }
}

std::ostream& operator<<(std::ostream& output, const InstanceReader& instanceReader) {
    output << "NAME: " << instanceReader._name << "\n" <<
        "TYPE: " << instanceReader._type << "\n" << 
        "COMMENT: " << instanceReader._comment << "\n" <<
        "DIMENSION: " << instanceReader._dimension << "\n" <<
        "METRIC: " << instanceReader._metric->code() << "\n";
    return output;
}

const HostMemoryInstance* InstanceReader::createHostMemoryInstance() const {
    return new HostMemoryInstance(this->_x, this->_y, this->_metric, this->_dimension);
}

InstanceReader::~InstanceReader() {
    for (int i = 0; i < this->METRIC_COUNT; i++) 
        delete this->METRICS[i];
    delete [] this->_x;
    delete [] this->_y;
}
