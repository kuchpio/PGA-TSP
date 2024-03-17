all: */*.cpp
	nvcc -x cu -std=c++11 main.cpp Instance/InstanceReader.cpp Instance/HostMemoryInstance.cpp Instance/GlobalMemoryInstance.cpp Metrics/Euclidean2D.cpp Metrics/CeilEuclidean2D.cpp -o pga-tsp

pull: 
	git pull https://kuchpio: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@github.com/kuchpio/PGA-TSP.git

clean:
	rm pga-tsp
