#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

EXECUTABLE="./x64/Release/PGA-TSP.exe"

COMMON_PARAMS="berlin52.tsp --islands 8 --population 256 --iterations 1000 --stalled-iterations 1000 --migrations 10 --stalled-migrations 10 --seed 100 --crossover 0.2 --mutation 0.2 --global"

for APPROACH in "--coarse-pmx" "--coarse-ox" "--fine --warps 32"; do
	PARAMS="$COMMON_PARAMS $APPROACH"
	echo "Running: $EXECUTABLE $PARAMS"
	$EXECUTABLE $PARAMS
	if [ $? -ne 0 ]; then
		echo "Error: The process terminated with a non-zero exit status."
		exit 1
	fi

	PARAMS="$COMMON_PARAMS --elitism $APPROACH"
	echo "Running: $EXECUTABLE $PARAMS"
	$EXECUTABLE $PARAMS
	if [ $? -ne 0 ]; then
		echo "Error: The process terminated with a non-zero exit status."
		exit 1
	fi
done

echo "All processes completed successfully."
