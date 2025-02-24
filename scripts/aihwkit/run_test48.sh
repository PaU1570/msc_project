#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_mp.sh"
ANALYSIS_SCRIPT_SOURCE_ASYM="${SCRIPT_DIR}/run_mnist_mp_multithreshold.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

for i in {1..3}; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/symmetric/zero_noise_run_$i" 25 0 0 0 0 0
    bash "$ANALYSIS_SCRIPT_SOURCE_ASYM" "$data_directory" "$output_directory/asymmetric/zero_noise_run_$i" 25 0 0 0 0 0
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/symmetric/default_noise_run_$i" 25 0.3 0.3 0.3 0.3 0.3
    bash "$ANALYSIS_SCRIPT_SOURCE_ASYM" "$data_directory" "$output_directory/asymmetric/default_noise_run_$i" 25 0.3 0.3 0.3 0.3 0.3
done
