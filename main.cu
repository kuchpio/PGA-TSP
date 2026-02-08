#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>

#include "args.hxx"
#include "Instance/InstanceReader.h"
#include "Instance/TextureMemoryInstance.h"
#include "Instance/GlobalMemoryInstance.h"
#include "Algorithm/FineGrained.h"
#include "Algorithm/CoarseGrained.h"
#include "Algorithm/OXAproch.h"

int main(int argc, char* argv[])
{
	args::ArgumentParser parser("This program uses parallel (CUDA) genetic algorithm to solve travelling salesman problem.", "Authors: Piotr Kucharczyk | Bartosz Maj.");
	args::HelpFlag helpFlag(parser, "help", "Display this help menu", { 'h', "help" });
	args::Group approachGroup(parser, "Approach:", args::Group::Validators::Xor);
	args::Flag coarsePMXFlag(approachGroup, "coarse-pmx", "Coarse grained approach with PMX crossover", { "coarse-pmx" });
	args::Flag coarseOXFlag(approachGroup, "coarse-ox", "Coarse grained approach with OX crossover", { "coarse-ox" });
	args::Flag fineFlag(approachGroup, "fine", "Fine grained approach", { "fine" });
	args::Group memoryGroup(parser, "Memory:", args::Group::Validators::Xor);
	args::Flag globalFlag(memoryGroup, "global", "Store instance in global memory", { "global" });
	args::Flag textureFlag(memoryGroup, "texture", "Store instance in texture memory", { "texture" });
	args::ValueFlag<unsigned int> islandsFlag(parser, "islands", "Number of islands", { "islands" }, 8);
	args::ValueFlag<unsigned int> populationFlag(parser, "population", "Population size of each island \n(ignored when --coarse-*)", { "population" }, 256);
	args::ValueFlag<unsigned int> iterationsFlag(parser, "iterations", "Number of iterations between migrations", { "iterations" }, 300);
	args::ValueFlag<unsigned int> migrationsFlag(parser, "migrations", "Number of migrations", { "migrations" }, 200);
	args::ValueFlag<unsigned int> stalledIterationsFlag(parser, "stalled-iterations", "Max number of consecutive iterations between migrations without fitness improvement", { "stalled-iterations" }, 100);
	args::ValueFlag<unsigned int> stalledMigrationsFlag(parser, "stalled-migrations", "Max number of consecutive migrations without fitness improvement on any island", { "stalled-migrations" }, 50);
	args::ValueFlag<float> crossoverProbabilityFlag(parser, "crossover", "Crossover probability", { "crossover" }, 0.5f);
	args::ValueFlag<float> mutationProbabilityFlag(parser, "mutation", "Mutation probability", { "mutation" }, 0.5f);
	args::Flag elitismFlag(parser, "elitism", "Enable elitism", { "elitism" });
	args::ValueFlag<unsigned int> warpCountFlag(parser, "warps", "Number of warps in block \n(ignored when --coarse-*)", { "warps" }, 16);
	args::ValueFlag<int> seedFlag(parser, "seed", "Seed for random number generator", { "seed" });
	args::Flag verboseFlag(parser, "verbose", "Print instance info, report progress", { "verbose" });
	args::Group requiredGroup(parser, "Required:", args::Group::Validators::All);
	args::Positional<std::string> inputFilename(requiredGroup, "file", "File that contains a travelling salesman problem instance description");
	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help)
	{
		std::cout << parser;
		return EXIT_SUCCESS;
	}
	catch (args::ParseError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return EXIT_FAILURE;
	}
	catch (args::ValidationError e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return EXIT_FAILURE;
	}

	std::ifstream input(args::get(inputFilename));
	if (!input.is_open()) {
		std::cerr << "Could not open file " << inputFilename << "\n";
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

	if (verboseFlag)
		std::cout << "INSTANCE SPECIFICATION\n" << instanceReader << "\n\n";

	tsp::IslandGeneticAlgorithmOptions options = {
		args::get(islandsFlag),
		args::get(populationFlag),
		args::get(iterationsFlag),
		args::get(migrationsFlag),
		args::get(crossoverProbabilityFlag),
		args::get(mutationProbabilityFlag),
		elitismFlag,
		args::get(stalledIterationsFlag),
		args::get(stalledMigrationsFlag)
	};
	int seed = seedFlag ? args::get(seedFlag) : (int)time(NULL);

	int* bestCycle = new int[globalMemoryInstance->size()];
	int bestCycleWeight;

	const auto start{ std::chrono::high_resolution_clock::now() };

	if (coarsePMXFlag) {
		if (globalFlag) {
			bestCycleWeight = tsp::solveTSPCoarseGrained(globalMemoryInstance->deviceInstance(), options, bestCycle, seed);
		}
		else {
			bestCycleWeight = tsp::solveTSPCoarseGrained(textureMemoryInstance->deviceInstance(), options, bestCycle, seed);
		}
	}
	else if (fineFlag) {
		if (globalFlag) {
			bestCycleWeight = tsp::solveTSPFineGrained(globalMemoryInstance->deviceInstance(), options, bestCycle, args::get(warpCountFlag), seed, verboseFlag);
		}
		else {
			bestCycleWeight = tsp::solveTSPFineGrained(textureMemoryInstance->deviceInstance(), options, bestCycle, args::get(warpCountFlag), seed, verboseFlag);
		}
	}
	else {
		if (globalFlag) {
			bestCycleWeight = tsp::solveTSPOXApproach(globalMemoryInstance->deviceInstance(), options, bestCycle, seed);
		}
		else {
			bestCycleWeight = tsp::solveTSPOXApproach(textureMemoryInstance->deviceInstance(), options, bestCycle, seed);
		}
	}

	const auto end{ std::chrono::high_resolution_clock::now() };

	if (bestCycleWeight >= 0 && verifyResults(hostInstance, bestCycle, bestCycleWeight))
		std::cout << "Best hamiltonian cycle length found: " << bestCycleWeight << ".\n";

	const auto executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Execution time: " << executionTime.count() << " ms.\n";

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