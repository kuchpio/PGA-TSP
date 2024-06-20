#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

COMMON_PARAMS="pcb442.tsp --islands 8 --population 256 --global --seed 100 --elitism --crossover 0.2 --mutation 0.2"

TOTAL_ITERATIONS=1024

for ((ITERATIONS = 2; ITERATIONS <= TOTAL_ITERATIONS; ITERATIONS *= 2)); do
	MIGRATIONS=$((TOTAL_ITERATIONS / ITERATIONS))
	ITER_PARAMS="--iterations $ITERATIONS --stalled-iterations $ITERATIONS --migrations $MIGRATIONS --stalled-migrations $MIGRATIONS"
	for APPROACH in "--coarse-pmx" "--coarse-ox" "--fine --warps 32"; do
		PARAMS="$COMMON_PARAMS $ITER_PARAMS $APPROACH"
		echo "Running: $EXECUTABLE $PARAMS"
		$EXECUTABLE $PARAMS
		if [ $? -ne 0 ]; then
			echo "Error: The process terminated with a non-zero exit status."
			exit 1
		fi
	done
done

echo "All processes completed successfully."
