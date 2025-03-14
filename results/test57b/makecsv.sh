#!/bin/bash

MAKE_CSV_SCRIPT='/scratch/msc24h18/msc_project/src/msc_project/utils/run_aihwkit_to_csv.py'
SUBDIR=$1

for dir in ${SUBDIR}/*; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        python $MAKE_CSV_SCRIPT $dir $SUBDIR/${name}.csv
    fi
done