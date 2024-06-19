#!/bin/bash

PROJECT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")"/.. ; pwd -P )
cd "$PROJECT_PATH"

# Define the executable and input file
EXECUTABLE="./x64/Release/PGA-TSP.exe"
INPUT_FILE="berlin52.tsp"

# Define the common parameters
COMMON_PARAMS="--coarse-pmx --global --seed 100 --crossover 0.9 --mutation 0.3 --stalled-iterations 100 --stalled-migrations 50"

# Define the parameter sets
PARAM_SETS=(
    "--iterations 100 --migrations 1"
    "--iterations 1000 --migrations 1"
    "--iterations 1000 --migrations 10"
)

# Loop through each parameter set and run the executable
for PARAMS in "${PARAM_SETS[@]}"; do
    echo "Running: $EXECUTABLE $INPUT_FILE $COMMON_PARAMS $PARAMS"
    $EXECUTABLE $INPUT_FILE $COMMON_PARAMS $PARAMS
    if [ $? -ne 0 ]; then
        echo "Error: The process terminated with a non-zero exit status."
        exit 1
    fi
done

echo "All processes completed successfully."