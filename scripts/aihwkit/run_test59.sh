#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/run_mnist_mp_nomod.sh"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="$2"

# granularity_vals=(0.01 0.1 0.2 0.3 0.4)
# batch_size_vals=(1 2 4 8 16 32 64)
granularity_vals=(0.005 0.05 0.2)
batch_size_vals=(2 3 4)

N_BATCH=4

for batch in "${batch_size_vals[@]}"; do
    ((i=i%N_BATCH)); ((i++==0)) && wait

    bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/t2_gran_0.4_batch_${batch}" 25 0 0 0 0 0 0.4 $batch &
done

# for gran in "${granularity_vals[@]}"; do
#     ((i=i%N_BATCH)); ((i++==0)) && wait
    
#     bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/gran_${gran}_batch_1" 25 0 0 0 0 0 $gran 1 &
# done

# for batch in "${batch_size_vals[@]}"; do
#     ((i=i%N_BATCH)); ((i++==0)) && wait

#     for gran in "${granularity_vals[@]}"; do        
#         bash "$ANALYSIS_SCRIPT_SOURCE" "$data_directory" "$output_directory/t2_gran_${gran}_batch_${batch}" 25 0 0 0 0 0 $gran $batch &
#     done
# done