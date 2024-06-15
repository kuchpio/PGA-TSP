#include <fstream>
#include <iostream>
#include <numeric>
#include "Instance/InstanceReader.h"
#include "Instance/TextureMemoryInstance.h"
#include "Instance/GlobalMemoryInstance.h"
#include "Algorithm/FineGrained.h"
#include "Algorithm/Basic.h"
#include "Algorithm/Kernels.h"

void usage(std::string programName) {
	std::cerr <<
		"USAGE: " << programName << " input\n" <<
		" input \tFile that contains a travelling salesman problem instance description.\n" <<
		"\tSupported formats are:\n" <<
		"\t - TSPLIB95 (see http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf)\n";
}

bool verifyResults(const tsp::IHostInstance* instance, unsigned short* bestCycle, unsigned int bestCycleWeight)
{
	unsigned int n = instance->size();
	bool* visited = new bool[n] { false };
	unsigned int verifiedCycleWeight = instance->edgeWeight(bestCycle[n - 1], bestCycle[0]);
	visited[bestCycle[n - 1]] = true;

	for (unsigned int i = 0; i < n - 1; i++) {
		if (visited[bestCycle[i]]) {
			std::cout << "VERIFICATION: Cycle is not hamiltonian. Vertex " << bestCycle[i] << " repeated.\n";
			delete[] visited;
			return false;
		}
		verifiedCycleWeight += instance->edgeWeight(bestCycle[i], bestCycle[i + 1]);
		visited[bestCycle[i]] = true;
	}

	if (bestCycleWeight != verifiedCycleWeight) {
		std::cout << "VERIFICATION: Cycle has different length (" << verifiedCycleWeight << ") than returned best cycle length (" << bestCycleWeight << ").\n";
		delete[] visited;
		return false;
	}

	delete[] visited;
	return true;
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

    unsigned short *bestCycle = new unsigned short[globalMemoryInstance->size()];
    tsp::IslandGeneticAlgorithmOptions options = {
    /* .islandCount: */             8,
    /* .islandPopulationSize: */    100,
    /* .isolatedIterationCount: */  50,
    /* .migrationCount: */          20,
    /* .crossoverProbability: */    0.5f,
    /* .mutationProbability: */     0.7f,
    /* .elitism: */                 true,
    /* .stalledMigrationsLimit: */  50
    };
    // int bestCycleWeight = tsp::solveTSPFineGrained(globalMemoryInstance->deviceInstance(), options, bestCycle, 28, 101, false);
	// int bestCycleWeight = tsp::solveTSP3(globalMemoryInstance->deviceInstance());
	// int bestCycleWeight = tsp::solveTSP4(globalMemoryInstance->deviceInstance());
	int bestCycleWeight = tsp::solveTSP5(globalMemoryInstance->deviceInstance(), 10, 1000, 500);

	std::cout << "\n";

	if (bestCycleWeight >= 0 /* && verifyResults(hostInstance, bestCycle, bestCycleWeight) */)
	    std::cout << "Best hamiltonian cycle length found: " << bestCycleWeight << "\n";

	// TODO: Save bestCycle to file

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