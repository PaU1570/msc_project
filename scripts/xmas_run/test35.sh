#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${PARENT_DIR}/aihwkit/run_mnist_mp_asymmetric.sh"
ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING="${PARENT_DIR}/aihwkit/run_mnist_mp_asymmetric_learnoutscaling.sh"
if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

write_noise_std_vals=($(seq 0 0.1 0.5))
for write_noise_std in "${write_noise_std_vals[@]}"; do
    bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/no_out_scaling_wnstd_up_down_${write_noise_std}_1_1" 25 0 0 "$write_noise_std" 0 0 1 1
    bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/no_out_scaling_wnstd_up_down_${write_noise_std}_1_2" 25 0 0 "$write_noise_std" 0 0 1 2
    bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/no_out_scaling_wnstd_up_down_${write_noise_std}_1_3" 25 0 0 "$write_noise_std" 0 0 1 3
    bash "${ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING}" "${data_directory}" "${output_directory}/learn_out_scaling_wnstd_up_down_${write_noise_std}_1_1" 25 0 0 "$write_noise_std" 0 0 1 1
    bash "${ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING}" "${data_directory}" "${output_directory}/learn_out_scaling_wnstd_up_down_${write_noise_std}_1_2" 25 0 0 "$write_noise_std" 0 0 1 2
    bash "${ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING}" "${data_directory}" "${output_directory}/learn_out_scaling_wnstd_up_down_${write_noise_std}_1_3" 25 0 0 "$write_noise_std" 0 0 1 3
done