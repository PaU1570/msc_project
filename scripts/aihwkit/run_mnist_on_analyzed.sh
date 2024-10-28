#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/mnist_simple.py"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="${2}/aihwkit"

if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

# find all Summary.dat files in the directory
files=$(find ${data_directory} -type f -name "*_Summary.dat")
for file in $files; do
    echo -e "\e[33m$file\e[0m"

    #if ! python ${ANALYSIS_SCRIPT_SOURCE} $file --output_dir ${output_directory} --epochs 25 --early_stopping ; then
    #    echo -e "\e[31mError\e[0m analyzing file"
    #else
    #    echo -e "\e[32mDone\e[0m"
    #fi
done

echo "All done!"