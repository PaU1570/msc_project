#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/mnist_simple.py"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory> [epochs (25)] [dw_min_dtod (0.3)] [dw_min_std (0.3)]"
    exit 1
fi

data_directory="$1"
output_directory="${2}/aihwkit"
epochs="${3:-25}"
dw_min_dtod="${4:-0.3}"
dw_min_std="${5:-0.3}"

echo "data_directory: $data_directory"
echo "output_directory: $output_directory"
echo "epochs: $epochs"
echo "dw_min_dtod: $dw_min_dtod"
echo "dw_min_std: $dw_min_std"

if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

# find all Summary.dat files in the directory
files=$(find ${data_directory} -type f -name "*_Summary.dat")
for file in $files; do
    echo -e "\e[33m$file\e[0m"

    if ! python ${ANALYSIS_SCRIPT_SOURCE} $file --output_dir ${output_directory} --epochs ${epochs} --dw_min_dtod ${dw_min_dtod} --dw_min_std ${dw_min_std} ; then
        echo -e "\e[31mError\e[0m analyzing file"
    else
        echo -e "\e[32mDone\e[0m"
    fi
done

# run python script to create csv file
python ${PARENT_DIR}/src/msc_project/utils/run_aihwkit_to_csv.py "${output_directory}" "${output_directory}"/$(basename $output_directory).csv

echo "All done!"