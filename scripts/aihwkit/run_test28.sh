#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE_LOS="${SCRIPT_DIR}/run_mnist_mp_learnoutscaling_adam.sh"
ANALYSIS_SCRIPT_SOURCE_NOS="${SCRIPT_DIR}/run_mnist_mp_adam.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

lr_vals=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4)
beta1_vals=(0.7 0.75 0.8 0.85 0.9)

N_BATCH=1

for lr in "${lr_vals[@]}"; do
    for beta1 in "${beta1_vals[@]}"; do
        ((i=i%N_BATCH)); ((i++==0)) && wait
        bash "$ANALYSIS_SCRIPT_SOURCE_NOS" "$data_directory" "$output_directory/no_out_scaling/lr_beta1_${lr}_${beta1}" 25 0 0 0 "$lr" "$beta1" 0.999 &
        bash "$ANALYSIS_SCRIPT_SOURCE_LOS" "$data_directory" "$output_directory/learn_out_scaling/lr_beta1_${lr}_${beta1}" 25 0 0 0 "$lr" "$beta1" 0.999 &
    done
done