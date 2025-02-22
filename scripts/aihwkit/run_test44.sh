#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_mp.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

noise_values=(0.1 0.15 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6)

for write_noise_std_mult in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/write_noise_std_${write_noise_std_mult}" 20 0.3 0.3 "$write_noise_std_mult"
done
