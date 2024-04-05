#ifndef __INSTANCE_READER_H__
#define __INSTANCE_READER_H__

#include <string>
#include <sstream>
#include <map>
#include <iostream>

#include "HostInstance.h"
#include "Metric.h"

namespace tsp {

	void trim(std::string& s) {
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
			return !std::isspace(c);
			}));
		s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {
			return !std::isspace(c);
			}).base(), s.end());
	}

	class InstanceReader {
	private:
		std::string _name, _type, _comment, _metric;
		int _dimension;
		float* _x, * _y;

	public:
		InstanceReader(std::istream& input) {
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
			this->_metric = keyValuePairs["EDGE_WEIGHT_TYPE"];
			this->_x = new float[this->_dimension];
			this->_y = new float[this->_dimension];

			// Read data
			while (input >> node >> x >> y) {
				this->_x[node - 1] = x;
				this->_y[node - 1] = y;
			}
		}

		const IHostInstance* createHostInstance() const {
			if (CeilEuclidean2D::IsMatching(this->_metric))
				return new HostMemoryInstance(this->_x, this->_y, this->_dimension, CeilEuclidean2D { });

			return new HostMemoryInstance(this->_x, this->_y, this->_dimension, Euclidean2D { });
		}

		template<typename DeviceInstance>
		const DeviceInstanceHostAdapter<DeviceInstance>* createDeviceInstance() const {
			if (CeilEuclidean2D::IsMatching(this->_metric))
				return new DeviceInstanceHostAdapter<DeviceInstance>(this->_x, this->_y, this->_dimension, CeilEuclidean2D { });

			return new DeviceInstanceHostAdapter<DeviceInstance>(this->_x, this->_y, this->_dimension, Euclidean2D { });
		}

		friend std::ostream& operator<<(std::ostream& output, const InstanceReader& instanceReader) {
			output << "NAME: " << instanceReader._name << "\n" <<
				"TYPE: " << instanceReader._type << "\n" <<
				"COMMENT: " << instanceReader._comment << "\n" <<
				"DIMENSION: " << instanceReader._dimension << "\n" <<
				"METRIC: " << instanceReader._metric;
			return output;
		}

		~InstanceReader() {
			delete[] this->_x;
			delete[] this->_y;
		}
	};
}

#endif