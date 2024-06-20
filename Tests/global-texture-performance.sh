#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

COMMON_PARAMS="d18512.tsp --islands 8 --population 256 --iterations 1000 --stalled-iterations 1000 --migrations 1 --stalled-migrations 1 --seed 100 --elitism --crossover 0.0 --mutation 0.0"

for APPROACH in "--coarse-pmx" "--coarse-ox" "--fine --warps 8" "--fine --warps 32"; do
	for MEMORY in "--global" "--texture"; do
		PARAMS="$COMMON_PARAMS $MEMORY $APPROACH"
		echo "Running: $EXECUTABLE $PARAMS"
		$EXECUTABLE $PARAMS
		if [ $? -ne 0 ]; then
			echo "Error: The process terminated with a non-zero exit status."
			exit 1
		fi
	done
done

echo "All processes completed successfully."
