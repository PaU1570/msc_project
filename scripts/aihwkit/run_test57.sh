#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_mp_asymmetric.sh"

if [ $# -ne 4 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory> <pulses up> <pulses down>"
    exit 1
fi

data_directory="$1"
output_directory="$2"
pulses_up="$3"
pulses_down="$4"

noise_values=(0 0.5 1 1.5 2 2.5 3)

for write_noise_std_mult in "${noise_values[@]}"; do
    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/${pulses_up}up${pulses_down}down/write_noise_std_${write_noise_std_mult}" 25 0 0 "$write_noise_std_mult" 0 0 "$pulses_up" "$pulses_down"
done
