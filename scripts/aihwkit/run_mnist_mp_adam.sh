#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/mnist_mixedprecision.py"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory> [epochs (25)] [dw_min_dtod (0.3)] [dw_min_std (0.3)] [write_noise_std_mult (0.0)] [lr (0.2)] [beta1 (0.9)] [beta2 (0.999)]"
    exit 1
fi

data_directory="$1"
output_directory="${2}/aihwkit"
epochs="${3:-25}"
dw_min_dtod="${4:-0.3}"
dw_min_std="${5:-0.3}"
write_noise_std_mult="${6:-0.0}"
lr="${7:-0.2}"
beta1="${8:-0.9}"
beta2="${9:-0.999}"

echo "data_directory: $data_directory"
echo "output_directory: $output_directory"
echo "epochs: $epochs"
echo "dw_min_dtod: $dw_min_dtod"
echo "dw_min_std: $dw_min_std"
echo "write_noise_std_mult: $write_noise_std_mult"
echo "lr: $lr"
echo "beta1: $beta1"
echo "beta2: $beta2"

if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

N_BATCH=4

# find all Summary.dat files in the directory
files=$(find ${data_directory} -type f -name "*_Summary.dat")
for file in $files; do
    ((i=i%N_BATCH)); ((i++==0)) && wait

    echo -e "\e[33m$file\e[0m"

    if ! python ${ANALYSIS_SCRIPT_SOURCE} $file \
                 --output_dir ${output_directory} \
                 --epochs ${epochs} \
                 --dw_min_dtod ${dw_min_dtod} --dw_min_std ${dw_min_std} --write_noise_std_mult ${write_noise_std_mult} \
                 --save_weights \
                 --optimizer adam --lr ${lr} --beta1 ${beta1} --beta2 ${beta2}; then
        echo -e "\e[31mError\e[0m analyzing file"
    else
        echo -e "\e[32mDone\e[0m"
    fi &
done

# run python script to create csv file
#python ${PARENT_DIR}/src/msc_project/utils/run_aihwkit_to_csv.py "${output_directory}" "${output_directory}"/$(basename $output_directory).csv

echo "All done!"
