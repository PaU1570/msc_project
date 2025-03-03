#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_sgd_onesided.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

noise_values=(0.1 0.2 0.3 0.4 0.5)

for dw_min_dtod in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/noise_dw_min_dtod_${dw_min_dtod}" 25 "$dw_min_dtod" 0 0 0 0
done

for dw_min_std in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/noise_dw_min_std_${dw_min_std}" 25 0 "$dw_min_std" 0 0 0
done

for w_min_dtod in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/noise_w_min_dtod_${w_min_dtod}" 25 0 0 0 "$w_min_dtod" 0
done

for w_max_dtod in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/noise_w_max_dtod_${w_max_dtod}" 25 0 0 0 0 "$w_max_dtod"
done