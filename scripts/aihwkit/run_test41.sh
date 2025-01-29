#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${PARENT_DIR}/aihwkit/run_mnist_sgd_adam_lr.sh"
if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

lr_vals=(0.01 0.05 0.1 0.2 0.4 0.6 1 2)
for lr in "${lr_vals[@]}"; do
    bash "${ANALYSIS_SCRIPT_SOURCE}" "${data_directory}" "${output_directory}/lr_${lr}" 25 0.3 0.3 0.0 0.3 0.3 "$lr"
done