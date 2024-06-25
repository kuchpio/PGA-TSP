#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

# INSTANCE="berlin52.tsp";
INSTANCE="pcb442.tsp";
COMMON_PARAMS="$INSTANCE --islands 8 --population 256 --iterations 100 --stalled-iterations 100 --migrations 10 --stalled-migrations 10 --global --seed 100 --elitism"

for APPROACH in "--coarse-pmx" "--coarse-ox" "--fine --warps 32"; do
	for CROSSOVER_PROBABILITY in 0.0 0.2 0.4 0.6 0.8 1.0; do
		for MUTATION_PROBABILITY in 0.0 0.2 0.4 0.6 0.8 1.0; do
			PARAMS="$COMMON_PARAMS --crossover $CROSSOVER_PROBABILITY --mutation $MUTATION_PROBABILITY $APPROACH"
			echo "Running: $EXECUTABLE $PARAMS"
			$EXECUTABLE $PARAMS
			if [ $? -ne 0 ]; then
				echo "Error: The process terminated with a non-zero exit status."
				exit 1
			fi
		done
	done
done

echo "All processes completed successfully."
