# Parallel genetic algorithm for TSP

This project is a multi-GPU CUDA implementation of a genetic algorithm for solving the Travelling Salesman Problem.

### Building

Before compilation, make sure you have CUDA and CUDA-aware MPI available in your system.
Also, remember to run
```
git submodule update --init --recursive
```
to fetch other necessary dependencies. 
The project can then be built using standard CMake procedures.

### Running

The input must be provided in the TSPLIB format, but currently only `EUC_2D` and `CEIL_2D` metrics are supported.
All options of the genetic algorithm and its execution can be found in the help message.
```
./pga-tsp --help
```
The program relies on CUDA-aware MPI communication for multi-GPU execution.
Each MPI process should correspond to one GPU.

### How does it work?

Currently, only the `--fine` approach supports multi-GPU execution.
It uses the entire warp to process each pair of chromosomes, in contrast to `--coarse*` approaches, which assign each pair to a single thread.
This improves memory coalescing and reduces the need for synchronization. 
As a result, during the execution, the `--fine` approach is more efficient in most cases.

The evolution (selection + mutation + crossover) progresses mostly within each island, which consists of all specimens processed on the same GPU block.
To pass the genetic information between islands, a migration is performed every several generations. 
The best specimen from each island replaces the worst ones from the cyclically next island located on the same continent (GPU).
To pass the genetic information between continents (GPUs), a supermigration is scheduled every several migrations.
All the best specimens from each continent replace all the worst ones from the cyclically next continent.
