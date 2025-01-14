#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${PARENT_DIR}/aihwkit/run_mnist_mp_learnoutscaling.sh"
if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

N_BATCH=1

# dw_min_dtod_vals=($(seq 0 0.05 0.5))
# for dw_min_dtod in "${dw_min_dtod_vals[@]}"; do
#     ((i=i%N_BATCH)); ((i++==0)) && wait
#     bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/dw_min_dtod_${dw_min_dtod}" 25 "$dw_min_dtod" 0 0 0 0 &
# done

# dw_min_std_vals=($(seq 0 0.05 0.5))
# for dw_min_std in "${dw_min_std_vals[@]}"; do
#     ((i=i%N_BATCH)); ((i++==0)) && wait
#     bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/dw_min_std_${dw_min_std}" 25 0 "$dw_min_std" 0 0 0 &
# done

write_noise_std_vals=($(seq 0 0.1 0.5))
for write_noise_std in "${write_noise_std_vals[@]}"; do
    ((i=i%N_BATCH)); ((i++==0)) && wait
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/write_noise_std_0_0_${write_noise_std}" 25 0 0 "$write_noise_std" 0 0 &
done

for write_noise_std in "${write_noise_std_vals[@]}"; do
    ((i=i%N_BATCH)); ((i++==0)) && wait
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/write_noise_std_0.3_0.3_${write_noise_std}" 25 0.3 0.3 "$write_noise_std" 0.3 0.3 &
done

# w_min_dtod_vals=($(seq 0 0.05 0.5))
# for w_min_dtod in "${w_min_dtod_vals[@]}"; do
#     ((i=i%N_BATCH)); ((i++==0)) && wait
#     bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/w_min_dtod_${w_min_dtod}" 25 0 0 0 "$w_min_dtod" 0 &
# done

# w_max_dtod_vals=($(seq 0 0.05 0.5))
# for w_max_dtod in "${w_max_dtod_vals[@]}"; do
#     ((i=i%N_BATCH)); ((i++==0)) && wait
#     bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/w_max_dtod_${w_max_dtod}" 25 0 0 0 0 "$w_max_dtod" &
# done