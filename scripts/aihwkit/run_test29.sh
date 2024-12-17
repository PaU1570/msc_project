#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_mp_learnoutscaling.sh"
if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

write_noise_std_mult_vals=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

N_BATCH=1

for wn_mult in "${write_noise_std_mult_vals[@]}"; do
    ((i=i%N_BATCH)); ((i++==0)) && wait
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/dtod_std_write_0.3_0.3_${wn_mult}" 25 0.3 0.3 "$wn_mult" 0.8 &
done