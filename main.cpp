#include <fstream>
#include <iostream>
#include <numeric>
#include "Instance/InstanceReader.h"
#include "Instance/TextureMemoryInstance.h"
#include "Instance/GlobalMemoryInstance.h"
#include "Algorithm/Basic.h"
#include "Algorithm/Kernels.h"
void usage(std::string programName) {
	std::cerr <<
		"USAGE: " << programName << " input\n" <<
		" input \tFile that contains a travelling salesman problem instance description.\n" <<
		"\tSupported formats are:\n" <<
		"\t - TSPLIB95 (see http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf)\n";
}

int main(int argc, char* argv[])
{
	std::string programName = argv[0];
	if (argc != 2) {
		usage(programName);
		return EXIT_FAILURE;
	}
	std::string inputFilename = argv[1];
	std::ifstream input(inputFilename);
	if (!input.is_open()) {
		std::cerr << programName << ": Could not open file " << inputFilename << "\n";
		usage(programName);
		return EXIT_FAILURE;
	}
	tsp::InstanceReader instanceReader(input);
	input.close();

	if (cudaSetDevice(0) != cudaSuccess) {
		std::cerr << "Could not set device. \n";
		return EXIT_FAILURE;
	}

	auto* hostInstance = instanceReader.createHostInstance();
	auto* globalMemoryInstance = instanceReader.createDeviceInstance<tsp::GlobalMemoryInstance>();
	auto* textureMemoryInstance = instanceReader.createDeviceInstance<tsp::TextureMemoryInstance>();

	int* canonicalCycle = new int[hostInstance->size()];
	std::iota(canonicalCycle, canonicalCycle + hostInstance->size(), 0);

	std::cout << "INSTANCE SPECIFICATION\n" << instanceReader << "\n";

	std::cout << "CANONICAL CYCLE TOTAL DISTANCE (host): " <<
		hostInstance->hamiltonianCycleWeight(canonicalCycle) << "\n";

	std::cout << "CANONICAL CYCLE TOTAL DISTANCE (device global memory): " <<
		globalMemoryInstance->hamiltonianCycleWeight(canonicalCycle) << "\n";

	std::cout << "CANONICAL CYCLE TOTAL DISTANCE (device texture memory): " <<
		textureMemoryInstance->hamiltonianCycleWeight(canonicalCycle) << "\n";

	int opt = tsp::solveTSP2(globalMemoryInstance->deviceInstance());

	std::cout << "Optimal hamiltonian cycle length found: " << opt << "\n";

	delete hostInstance;
	delete globalMemoryInstance;
	delete textureMemoryInstance;
	delete[] canonicalCycle;

	if (cudaDeviceReset() != cudaSuccess) {
		std::cerr << "Could not reset device. \n";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}