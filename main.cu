#include <fstream>
#include <iostream>
#include <chrono>

#include <mpi.h>

#include "args.hxx"
#include "Instance/InstanceReader.h"
#include "Instance/TextureMemoryInstance.h"
#include "Instance/GlobalMemoryInstance.h"
#include "Algorithm/FineGrained.h"
#include "Algorithm/CoarseGrained.h"
#include "Algorithm/OXAproch.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

	int mpiRank, mpiSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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
	args::ValueFlag<unsigned int> intercontinentalMigrationsPeriodFlag(parser, "superemigration-period", "Number of migrations between consecutive inter-GPU migrations", { "superemigration-period" }, 10);
	args::ValueFlag<unsigned int> stalledIterationsFlag(parser, "stalled-iterations", "Max number of consecutive iterations between migrations without fitness improvement", { "stalled-iterations" }, 100);
	args::ValueFlag<unsigned int> stalledMigrationsFlag(parser, "stalled-migrations", "Max number of consecutive migrations without fitness improvement on any island", { "stalled-migrations" }, 50);
	args::ValueFlag<float> crossoverProbabilityFlag(parser, "crossover", "Crossover probability", { "crossover" }, 0.5f);
	args::ValueFlag<float> mutationProbabilityFlag(parser, "mutation", "Mutation probability", { "mutation" }, 0.5f);
	args::Flag elitismFlag(parser, "elitism", "Enable elitism", { "elitism" });
	args::ValueFlag<unsigned int> warpCountFlag(parser, "warps", "Number of warps in block \n(ignored when --coarse-*)", { "warps" }, 16);
	args::ValueFlag<int> seedFlag(parser, "seed", "Seed for random number generator", { "seed" });
	args::Flag verboseFlag(parser, "verbose", "Print instance info, report progress", { "verbose" });
	args::ValueFlag<std::string> historyFlag(parser, "history", "History pathname base", { "history" });
	args::Group requiredGroup(parser, "Required:", args::Group::Validators::All);
	args::Positional<std::string> inputFilename(requiredGroup, "file", "File that contains a travelling salesman problem instance description");
	args::Positional<std::string> outputFilename(parser, "tour", "Output file with solution");

	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (const args::Help&)
	{
		if (mpiRank == 0) std::cout << parser;
		MPI_Finalize();
		return EXIT_SUCCESS;
	}
	catch (const args::ParseError& e)
	{
		if (mpiRank == 0) std::cerr << e.what() << std::endl << parser;
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	catch (const args::ValidationError& e)
	{
		if (mpiRank == 0) std::cerr << e.what() << std::endl << parser;
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	if (coarsePMXFlag || coarseOXFlag) {
		if (mpiRank == 0) std::cerr << "Coarse approaches currently not supported. \n";
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	std::ifstream input(args::get(inputFilename));
	if (!input.is_open()) {
		std::cerr << "Could not open file " << inputFilename << "\n";
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	tsp::InstanceReader instanceReader(input);
	input.close();

	if (verboseFlag && mpiRank == 0)
		std::cout << "INSTANCE SPECIFICATION\n" << instanceReader << "\n\n";

	int deviceCount;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		std::cerr << "Could not set device. \n";
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	if (cudaSetDevice(mpiRank % deviceCount) != cudaSuccess) {
		std::cerr << "Could not set device. \n";
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	auto* hostInstance = instanceReader.createHostInstance();
	auto* globalMemoryInstance = instanceReader.createDeviceInstance<tsp::GlobalMemoryInstance>();
	auto* textureMemoryInstance = instanceReader.createDeviceInstance<tsp::TextureMemoryInstance>();

	tsp::IslandGeneticAlgorithmOptions options = {
		args::get(islandsFlag),
		args::get(populationFlag),
		args::get(iterationsFlag),
		args::get(migrationsFlag),
		args::get(intercontinentalMigrationsPeriodFlag),
		args::get(crossoverProbabilityFlag),
		args::get(mutationProbabilityFlag),
		elitismFlag,
		args::get(stalledIterationsFlag),
		args::get(stalledMigrationsFlag)
	};
	int seed = mpiRank + (seedFlag ? args::get(seedFlag) : static_cast<int>(time(nullptr)));

	std::vector<int> bestCycle(globalMemoryInstance->size());
	int bestCycleWeightAndRank[2];
    bestCycleWeightAndRank[1] = mpiRank;

	const auto start{ std::chrono::high_resolution_clock::now() };

	if (coarsePMXFlag) {
		if (globalFlag) {
			bestCycleWeightAndRank[0] = tsp::solveTSPCoarseGrained(globalMemoryInstance->deviceInstance(), options, bestCycle.data(), seed);
		}
		else {
			bestCycleWeightAndRank[0] = tsp::solveTSPCoarseGrained(textureMemoryInstance->deviceInstance(), options, bestCycle.data(), seed);
		}
	}
	else if (fineFlag) {
		if (globalFlag) {
			bestCycleWeightAndRank[0] = tsp::solveTSPFineGrained(globalMemoryInstance->deviceInstance(), options, bestCycle.data(), args::get(warpCountFlag), mpiRank, mpiSize, MPI_INT, seed, args::get(historyFlag), verboseFlag);
		}
		else {
			bestCycleWeightAndRank[0] = tsp::solveTSPFineGrained(textureMemoryInstance->deviceInstance(), options, bestCycle.data(), args::get(warpCountFlag), mpiRank, mpiSize, MPI_INT, seed, args::get(historyFlag), verboseFlag);
		}
	}
	else {
		if (globalFlag) {
			bestCycleWeightAndRank[0] = tsp::solveTSPOXApproach(globalMemoryInstance->deviceInstance(), options, bestCycle.data(), seed);
		}
		else {
			bestCycleWeightAndRank[0] = tsp::solveTSPOXApproach(textureMemoryInstance->deviceInstance(), options, bestCycle.data(), seed);
		}
	}

	int globalBestCycleWeightAndRank[2];
	MPI_Allreduce(bestCycleWeightAndRank, globalBestCycleWeightAndRank, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
	if (0 != globalBestCycleWeightAndRank[1]) {
		if (mpiRank == globalBestCycleWeightAndRank[1])
			MPI_Send(bestCycle.data(), bestCycle.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);

		if (mpiRank == 0)
			MPI_Recv(bestCycle.data(), bestCycle.size(), MPI_INT, globalBestCycleWeightAndRank[1], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	const auto end{ std::chrono::high_resolution_clock::now() };

	if (mpiRank == 0 && globalBestCycleWeightAndRank[0] >= 0 && verifyResults(hostInstance, bestCycle.data(), globalBestCycleWeightAndRank[0]))
		std::cout << "Best hamiltonian cycle length found: " << globalBestCycleWeightAndRank[0] << " on [" << globalBestCycleWeightAndRank[1] << "].\n";

	const auto executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "[" << mpiRank << "] Execution time: " << executionTime.count() << " ms.\n";

	if (mpiRank == 0 && !args::get(outputFilename).empty()) {
		std::ofstream output(args::get(outputFilename));
		if (output.is_open()) {
			std::cout << "Saving output to " << args::get(outputFilename) << "\n";

			for (unsigned int i = 0; i < globalMemoryInstance->size(); i++)
				output << bestCycle[i] << "\n";

			output.close();
		} else {
			std::cerr << "Could not open file " << args::get(outputFilename) << "\n";
		}
	}

	delete hostInstance;
	delete globalMemoryInstance;
	delete textureMemoryInstance;

	MPI_Finalize();

	if (cudaDeviceReset() != cudaSuccess) {
		std::cerr << "Could not reset device. \n";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
