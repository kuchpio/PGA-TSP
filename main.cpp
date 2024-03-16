#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include "Instance/IInstance.h"
#include "Instance/InstanceReader.h"

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
    InstanceReader instanceReader(input);                                                                                                           
    input.close();

    const IInstance *instance = instanceReader.createHostMemoryInstance();
    int *canonicalCycle = new int[instance->size()];
    std::iota(canonicalCycle, canonicalCycle + instance->size(), 0);

    std::cout << "INSTANCE SPECIFICATION\n" << instanceReader << "\n";
    std::cout << "CANONICAL CYCLE TOTAL DISTANCE: " << 
        instance->hamiltonianCycleWeight(canonicalCycle) << "\n";

    delete instance;
    delete[] canonicalCycle;
	return EXIT_SUCCESS;
}
