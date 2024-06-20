#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

COMMON_PARAMS="pcb442.tsp --islands 8 --population 256 --iterations 512 --migrations 512 --global --crossover 0.2 --mutation 0.2 --seed 100 --elitism"

for APPROACH in "--fine --warps 32"; do
	for STALLED_ITER in 512 256 128 64 32; do
		for STALLED_MIGR in 512 256 128 64 32; do
			PARAMS="$COMMON_PARAMS --stalled-iterations $STALLED_ITER --stalled-migrations $STALLED_MIGR $APPROACH"
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
