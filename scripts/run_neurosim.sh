#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
NEUROSIM_DIR="${PARENT_DIR}/MLP_NeuroSim_V3.0/"
WORKING_DIR=$(pwd)

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output_directory>"
    exit 1
fi

directory="$1"
output_directory="$2"
# Create the output directory if it doesn't exist
if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

error_files="${output_directory}/error_files.txt"

# find all json files in the directory
files=$(find ${directory} -type f -name "*.json")
for file in $files; do
    summary_file="$(dirname $file)/$(basename $file .json)_Summary.dat"
    output_file="${output_directory}/$(basename $file .json)_output.dat"
    if [ -f "$summary_file" ]; then
        echo -e "\e[33m$file\e[0m"
        head -n 6 "$summary_file" >> "$output_file"
    else
        echo -e "\e[33mWarning\e[0m: Summary file for $file not found. Output will be incomplete."
    fi
    echo "NeuroSim output:" >> "$output_file"
    echo "=================" >> "$output_file"
    if ! (cd ${NEUROSIM_DIR}; ./main $WORKING_DIR/$file >> "${WORKING_DIR}/${output_file}") ; then
        echo -e "\e[31mError\e[0m analyzing file (see ${error_files})"
        echo "$file" >> ${error_files}
    else
        echo -e "\e[32mDone\e[0m"
    fi
done