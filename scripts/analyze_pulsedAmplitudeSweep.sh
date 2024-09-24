#!/bin/bash

SCRIPT_SOURCE="/scratch/msc24h18/MLP_NeuroSim_V3.0/nonlinear_fit.py"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory="$1"
echo "Analyzing data in ${directory}"

# Find all files in the directory that contain the string "pulsedAmplitudeSweep"
files=$(find ${directory} -type f -name "*pulsedAmplitudeSweep*.csv" | grep -v "(ALL)")
for file in $files; do
    echo "$file"
    python ${SCRIPT_SOURCE} $file saveplot > /dev/null
done

echo "Done."

