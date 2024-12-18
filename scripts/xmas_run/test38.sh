#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output directory>"
    exit 1
fi

output_directory="$1"

lr_vals=($(seq 0.05 0.05 1.0))
for lr in "${lr_vals[@]}"; do
    python ${PARENT_DIR}/fp_mnist.py \
    --output_dir "$output_directory/lr_${lr}" \
    --epochs 25 \
    --lr "$lr" \
    --save_weights
done