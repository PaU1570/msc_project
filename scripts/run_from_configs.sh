#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
WORKING_DIR=$(pwd)

if [ $# -ne 3 ]; then
    echo "Usage: $0 <configs directory> <output directory> <number of runs>"
    exit 1
fi

config_directory="$1"
output_directory="$2"
num_runs="$3"

for ((i=1; i<=$num_runs; i++)); do
    run_directory="$output_directory/run_$i"
    if [ ! -d "$run_directory" ]; then
        mkdir -p "$run_directory"
    fi

    ${SCRIPT_DIR}/run_neurosim.sh "${config_directory}" "${run_directory}/neurosim"
    # run python script to collect results into one csv
    python ${PARENT_DIR}/src/msc_project/utils/run_neurosim_to_csv.py "${run_directory}/neurosim" "${output_directory}"/$(basename $run_directory).csv
    echo "Run $i done!"
done
echo "All done!"