#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

COMMON_PARAMS="berlin52.tsp --islands 8 --population 256 --iterations 10000 --stalled-iterations 10000 --migrations 1 --stalled-migrations 1 --global --seed 100 --elitism"

for CROSSOVER_PROBABILITY in 0.0 1.0; do
    for MUTATION_PROBABILITY in 0.0 1.0; do
	    for APPROACH in "--coarse-pmx" "--coarse-ox" "--fine --warps 8" "--fine --warps 32"; do
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
