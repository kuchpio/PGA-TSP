#ifndef __METRIC_H__
#define __METRIC_H__

#include <cuda_runtime.h>
#include <string>
#include <cmath>

namespace tsp {

	typedef struct Eucludean2D {
	public:
		static inline bool IsMatching(const std::string& code) {
			return code == "EUC_2D";
		}
	} Euclidean2D;

	typedef struct CeilEucludean2D {
	public: 
		static inline bool IsMatching(const std::string& code) {
			return code == "CEIL_2D";
		}
	} CeilEuclidean2D;

	typedef struct Geolocation {
	public:
		static inline bool IsMatching(const std::string& code) {
			return code == "GEOM";
		}
	} Geolocation;

	__device__ __host__
	inline int distance(Euclidean2D metric, float x1, float y1, float x2, float y2) {
		float dx = x1 - x2;
		float dy = y1 - y2;
		return (int)roundf(sqrtf(dx * dx + dy * dy));
	}

	__device__ __host__
	inline int distance(CeilEuclidean2D metric, float x1, float y1, float x2, float y2) {
		float dx = x1 - x2;
		float dy = y1 - y2;
		return (int)ceilf(sqrtf(dx * dx + dy * dy));
	}

	__device__ __host__
	inline int distance(Geolocation metric, float lat1, float lng1, float lat2, float lng2) {
		float lat1_rad = M_PI * lat1 / 180.0f;
		float lat2_rad = M_PI * lat2 / 180.0f;
		float lng_diff_rad = M_PI * (lng1 - lng2) / 180.0f;

		float q1 = cosf(lat2_rad) * sinf(lng_diff_rad);
		float q3 = sinf(lng_diff_rad / 2.0f);
		float q4 = cosf(lng_diff_rad / 2.0f);
		float q2 = sinf(lat1_rad + lat2_rad) * q3 * q3 - sinf(lat1_rad - lat2_rad) * q4 * q4;
		float q5 = cosf(lat1_rad - lat2_rad) * q4 * q4 - cosf(lat1_rad + lat2_rad) * q3 * q3;

		return (int)(6378388.0f * atan2f(sqrtf(q1*q1 + q2*q2), q5) + 1.0f);
	}

}

#endif
