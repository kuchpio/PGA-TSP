#include <fstream>
#include <iostream>
#include <numeric>
#include "Instance/InstanceReader.h"
#include "Instance/TextureMemoryInstance.h"
#include "Instance/GlobalMemoryInstance.h"
#include "Algorithm/FineGrained.h"

void usage(std::string programName) {
    std::cerr << 
        "USAGE: " << programName << " input\n" << 
        " input \tFile that contains a travelling salesman problem instance description.\n" << 
        "\tSupported formats are:\n" <<
        "\t - TSPLIB95 (see http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf)\n";
}

int main(int argc, char *argv[])
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

    auto *hostInstance = instanceReader.createHostInstance();
    auto *globalMemoryInstance = instanceReader.createDeviceInstance<tsp::GlobalMemoryInstance>();
    auto *textureMemoryInstance = instanceReader.createDeviceInstance<tsp::TextureMemoryInstance>();

    int *canonicalCycle = new int[hostInstance->size()];
    std::iota(canonicalCycle, canonicalCycle + hostInstance->size(), 0);

    std::cout << "INSTANCE SPECIFICATION\n" << instanceReader << "\n";

    std::cout << "CANONICAL CYCLE TOTAL DISTANCE (host): " << 
        hostInstance->hamiltonianCycleWeight(canonicalCycle) << "\n";

    std::cout << "CANONICAL CYCLE TOTAL DISTANCE (device global memory): " << 
        globalMemoryInstance->hamiltonianCycleWeight(canonicalCycle) << "\n";

    std::cout << "CANONICAL CYCLE TOTAL DISTANCE (device texture memory): " << 
        textureMemoryInstance->hamiltonianCycleWeight(canonicalCycle) << "\n";


    unsigned short *optimalCycle = new unsigned short[globalMemoryInstance->size()];
    tsp::IslandGeneticAlgorithmOptions options = {
        8,      // .islandCount
        100,     // .islandPopulationSize
        100,     // .isolatedIterationCount
        10,     // .migrationCount
        0.5f,    // .crossoverProbability
		0.7f,    // .mutationProbability
		true,    // .elitism
        10,   // .stableMigrationCount
    };
    int opt = tsp::solveTSPFineGrained(globalMemoryInstance->deviceInstance(), options, optimalCycle, 32, 101, true);

    std::cout << "\n\nOptimal hamiltonian cycle length found: " << opt << "\n";

    // Verification
    {
        unsigned int n = globalMemoryInstance->size();
        bool* visited = new bool[n] { false };
        unsigned int verifiedCycleWeight = hostInstance->edgeWeight(optimalCycle[n - 1], optimalCycle[0]);
        visited[optimalCycle[n - 1]] = true;

		for (unsigned int i = 0; i < n - 1; i++) {
            if (visited[optimalCycle[i]]) {
                std::cout << "Vertex " << optimalCycle[i] << " repeated.\n";
            }
			verifiedCycleWeight += hostInstance->edgeWeight(optimalCycle[i], optimalCycle[i + 1]);
            visited[optimalCycle[i]] = true;
		}

        std::cout << "\n\nOptimal hamiltonian cycle length verified: " << verifiedCycleWeight << "\n";

        delete[] visited;
    }

    delete hostInstance;
    delete globalMemoryInstance;
    delete textureMemoryInstance;
    delete[] canonicalCycle;
    delete[] optimalCycle;

    if (cudaDeviceReset() != cudaSuccess) {
        std::cerr << "Could not reset device. \n";
        return EXIT_FAILURE;
    }

	return EXIT_SUCCESS;
}
