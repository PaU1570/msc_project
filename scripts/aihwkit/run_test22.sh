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

dw_min_dtod_vals=(0 0.1 0.2 0.3)
dw_min_std_vals=(0 0.1 0.2 0.3)
write_noise_std_mult_vals=(0 0.1 0.2 0.3)

N_BATCH=4

for dw_min_dtod in "${dw_min_dtod_vals[@]}"; do
    for dw_min_std in "${dw_min_std_vals[@]}"; do
        for write_noise_std_mult in "${write_noise_std_mult_vals[@]}"; do
            ((i=i%N_BATCH)); ((i++==0)) && wait
            bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/dtod_std_write_${dw_min_dtod}_${dw_min_std}_${write_noise_std_mult}" 25 "$dw_min_dtod" "$dw_min_std" "$write_noise_std_mult" &
        done
    done
done