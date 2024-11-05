#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_mp_on_analyzed.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

noise_values=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

for dw_min_std in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/dw_min_std_${dw_min_std}" 25  0.3 "$dw_min_std" 0.0
done