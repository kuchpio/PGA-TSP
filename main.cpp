#include <fstream>
#include <iostream>
#include <numeric>
#include "Instance/InstanceReader.h"
#include "Instance/TextureMemoryInstance.h"
#include "Instance/GlobalMemoryInstance.h"
#include "Algorithm/FineGrained.h"
#include "Algorithm/CoarseGrained.h"

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

	std::cout << "INSTANCE SPECIFICATION\n" << instanceReader << "\n\n";

    tsp::IslandGeneticAlgorithmOptions options = {
    /* .islandCount: */						2,
    /* .islandPopulationSize: */			256, // Ignored for CoarseGrained
    /* .isolatedIterationCount: */			1000,
    /* .migrationCount: */					10,
    /* .crossoverProbability: */			0.9f,
    /* .mutationProbability: */				0.3f,
    /* .elitism: */							true,
	/* .stalledIsolatedIterationsLimit */	500,
    /* .stalledMigrationsLimit: */			50
    };

    // unsigned short *bestCycle = new unsigned short[globalMemoryInstance->size()];
    // int bestCycleWeight = tsp::solveTSPFineGrained(globalMemoryInstance->deviceInstance(), options, bestCycle, 28, 101, true);

    int *bestCycle = new int[globalMemoryInstance->size()];
    int bestCycleWeight = tsp::solveTSPCoarseGrained(globalMemoryInstance->deviceInstance(), options, bestCycle, 102);

	std::cout << "\n";

	if (bestCycleWeight >= 0 && verifyResults(hostInstance, bestCycle, bestCycleWeight))
	    std::cout << "Best hamiltonian cycle length found: " << bestCycleWeight << "\n";

    delete hostInstance;
    delete globalMemoryInstance;
    delete textureMemoryInstance;
    delete[] bestCycle;

	if (cudaDeviceReset() != cudaSuccess) {
		std::cerr << "Could not reset device. \n";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}