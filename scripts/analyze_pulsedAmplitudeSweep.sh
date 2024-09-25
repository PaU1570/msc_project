#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${PARENT_DIR}/MLP_NeuroSim_V3.0/nonlinear_fit.py"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory="$1"
echo "Analyzing data in ${directory}"

# Find all csv files in the directory that contain the string "pulsedAmplitudeSweep" and don't contain the string "(ALL)"
files=$(find ${directory} -type f -name "*pulsedAmplitudeSweep*.csv" | grep -v "(ALL)")
for file in $files; do
    echo "$file"
    #python ${ANALYSIS_SCRIPT_SOURCE} $file saveplot > /dev/null
done

echo "Done."

