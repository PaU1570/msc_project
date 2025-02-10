#!/bin/zsh
FIGURES_DIR="/scratch/msc24h18/msc_project/results/report_figures/"
RESULTS_DIR="/scratch/msc24h18/msc_project/results/"
PLOT_SCRIPT="/scratch/msc24h18/msc_project/src/msc_project/utils/plot_utils.py"

FILENAME="neurosim_accuracy_numlevels.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test33/run_1.csv" \
    -x num_LTP_plus_LTD -y accuracy --hue device_id \
    --xlabel "Number of conductance levels (LTP + LTD)" --ylabel "Accuracy" --huelabel "Device ID" --title "NeuroSim Accuracy vs. Number of Conductance Levels" \
    --savefig "${FIGURES_DIR}/${FILENAME}"
echo "Generated figure: ${FILENAME}"


