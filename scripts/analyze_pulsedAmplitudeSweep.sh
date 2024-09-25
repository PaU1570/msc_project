#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${PARENT_DIR}/MLP_NeuroSim_V3.0/nonlinear_fit.py"

if [ $# -ne 3 ]; then
    echo "Usage: $0 <directory> <output_directory> <config_file>"
    exit 1
fi

directory="$1"
output_directory="$2"
# Create the output directory if it doesn't exist
if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

cp "$3" "${output_directory}/ref-config.json"
config_file="${output_directory}/ref-config.json"

error_files="${output_directory}/error_files.txt"
error_log="${output_directory}/error_log.log"

echo "Analyzing data in ${directory}"


# Find all csv files in the directory that contain the string "pulsedAmplitudeSweep" and don't contain the string "(ALL)"
files=$(find ${directory} -type f -name "*pulsedAmplitudeSweep*.csv" | grep -v "(ALL)")
for file in $files; do
    echo -e "\e[33m$file\e[0m"
    file_output_dir="${output_directory}/$(dirname $file)"
    if [ ! -d "$file_output_dir" ]; then
        mkdir -p "$file_output_dir"
    fi

    if ! python ${ANALYSIS_SCRIPT_SOURCE} $file --plotmode saveplot --configref ${config_file} --summary --dest ${file_output_dir} 2>> ${error_log} ; then
        echo -e "\e[31mError\e[0m analyzing file (see ${error_files} and ${error_log})"
        echo "$file" >> ${error_files}
    else
        echo -e "\e[32mDone\e[0m"
    fi
done


