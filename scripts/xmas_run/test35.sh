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

noise_vals=(0 0.3)
for noise in "${noise_vals[@]}"; do
    bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/no_out_scaling_noise_up_down_${noise}_1_1" 25 "$noise" "$noise" "$noise" "$noise" "$noise" 1 1
    bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/no_out_scaling_noise_up_down_${noise}_1_2" 25 "$noise" "$noise" "$noise" "$noise" "$noise" 1 2
    #bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/no_out_scaling_noise_up_down_${noise}_1_3" 25 "$noise" "$noise" "$noise" "$noise" "$noise" 1 3
    bash "${ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING}" "${data_directory}" "${output_directory}/learn_out_scaling_noise_up_down_${noise}_1_1" 25 "$noise" "$noise" "$noise" "$noise" "$noise" 1 1
    bash "${ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING}" "${data_directory}" "${output_directory}/learn_out_scaling_noise_up_down_${noise}_1_2" 25 "$noise" "$noise" "$noise" "$noise" "$noise" 1 2
    #bash "${ANALYSIS_SCRIPT_SOURCE_LEARNOUTSCALING}" "${data_directory}" "${output_directory}/learn_out_scaling_noise_up_down_${noise}_1_3" 25 "$noise" "$noise" "$noise" "$noise" "$noise" 1 3
done