#ifndef __METRIC_H__
#define __METRIC_H__

#include <cuda_runtime.h>
#include <string>

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

}

#endif
