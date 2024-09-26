#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
WORKING_DIR=$(pwd)

if [ $# -ne 3 ]; then
    echo "Usage: $0 <data directory> <output directory> <config_file>"
    exit 1
fi

data_directory="$1"
output_directory="$2"
config_file="$3"

# Create the output directory if it doesn't exist
if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

# run nonlinear fit analysis script
${SCRIPT_DIR}/analyze_pulsedAmplitudeSweep.sh $data_directory $output_directory $config_file
${SCRIPT_DIR}/run_neurosim.sh "${output_directory}/${data_directory}" "${output_directory}/neurosim"

# run python script to collect results into one csv
python ${PARENT_DIR}/utils/src/run_neurosim_to_csv.py "${output_directory}/neurosim" "${output_directory}"/$(basename $output_directory).csv
echo "All done!"