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

FILENAME="neurosim_accuracy_pulsewidth.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test33/run_1.csv" \
    -x pulseWidth -y accuracy --hue device_id \
    --scale lin-log --filter device_id ID301XR1000Oct ID161ZR15000 ID170ZR5000 ID181ZR1000 --fmt "o-" \
    --xlabel "Pulse Width (s)" --ylabel "Accuracy" --huelabel "Device ID" --title "NeuroSim Accuracy vs. Pulse Width" \
    --savefig "${FIGURES_DIR}/${FILENAME}"
echo "Generated figure: ${FILENAME}"

FILENAME="aihwkit_accuracy_dw.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test46/symmetric.csv" \
    -x granularity_up -y val_acc --hue device_id \
    --ylim 0.94 0.975 \
    --xlabel "Max ($\Delta w_0^u$, $\Delta w_0^d$)" --ylabel "Accuracy" --huelabel "Device ID" --title "AIHWKit Accuracy vs. $\Delta w_0$" \
    --savefig "${FIGURES_DIR}/${FILENAME}"
echo "Generated figure: ${FILENAME}"

FILENAME="aihwkit_ID170ZR5000_accuracy_energy.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x energy -y val_acc --filter device_id ID170ZR5000 --hue pulseWidth \
    --huescale log --scale lin-log \
    --xlabel "Energy/Area (upper bound) [\$J/cm^2\$]" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID170ZR5000 Accuracy vs. Energy"