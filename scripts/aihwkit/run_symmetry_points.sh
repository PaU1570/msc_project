#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ANALYSIS_SCRIPT_SOURCE="${SCRIPT_DIR}/symmetry_point.py"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <analyzed data directory> <output directory>"
    exit 1
fi

data_directory="$1"
output_directory="${2}"

if [ ! -d "$output_directory" ]; then
    mkdir -p "$output_directory"
fi

N_BATCH=24

# find all Summary.dat files in the directory
files=$(find ${data_directory} -type f -name "*_Summary.dat")
for file in $files; do
    ((i=i%N_BATCH)); ((i++==0)) && wait

    output_dirname=${output_directory}/$(basename $file _Summary.dat)
    output=${output_dirname}/symmetry_point.png

    if [ ! -d "$output_dirname" ]; then
        mkdir -p "$output_dirname"
    fi

    if python3 ${ANALYSIS_SCRIPT_SOURCE} $file --output ${output} ; then
        echo -e "\e[32mSaved plot to:\e[0m ${output}"
    else
        echo -e "\e[31mError analyzing file:\e[0m $file"
    fi &
    
done
wait
echo "All done!"