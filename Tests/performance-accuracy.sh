#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

INSTANCE="berlin52.tsp";
COMMON_PARAMS="$INSTANCE --islands 8 --population 256 --stalled-iterations 1000 --stalled-migrations 1000 --global --seed 100"

for APPROACH in "--coarse-pmx" "--coarse-ox" "--fine --warps 8" "--fine --warps 32"; do
	for ITERATIONS in 128 256 512; do
		for MIGRATIONS in 32 64 128; do
			for CROSSOVER_PROBABILITY in 0.25 0.5 0.75; do
				for MUTATION_PROBABILITY in 0.25 0.5 0.75; do
					PARAMS="$COMMON_PARAMS --iterations $ITERATIONS --migrations $MIGRATIONS --crossover $CROSSOVER_PROBABILITY --mutation $MUTATION_PROBABILITY $APPROACH"
					echo $APPROACH
					$EXECUTABLE $PARAMS
					if [ $? -ne 0 ]; then
						echo "Error: The process terminated with a non-zero exit status."
						exit 1
					fi
					PARAMS="$COMMON_PARAMS --iterations $ITERATIONS --migrations $MIGRATIONS --crossover $CROSSOVER_PROBABILITY --mutation $MUTATION_PROBABILITY $APPROACH --elitism"
					echo $APPROACH
					$EXECUTABLE $PARAMS
					if [ $? -ne 0 ]; then
						echo "Error: The process terminated with a non-zero exit status."
						exit 1
					fi
				done
			done
		
		done
	done

done

echo "All processes completed successfully."
