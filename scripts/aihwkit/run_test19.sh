#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

noise_values=(0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 2.1)

for dw_min_dtod in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/noise_${dw_min_dtod}_0.3" 25 "$dw_min_dtod" 0.3
done

for dw_min_std in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/noise_0.3_${dw_min_std}" 25  0.3 "$dw_min_std"
done