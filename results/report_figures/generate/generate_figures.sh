#!/bin/zsh
DATA_DIR="/scratch/msc24h18/msc_project/data/"
FIGURES_DIR="/scratch/msc24h18/msc_project/results/report_figures/"
RESULTS_DIR="/scratch/msc24h18/msc_project/results/"
PLOT_SCRIPT="/scratch/msc24h18/msc_project/src/msc_project/utils/plot_utils.py"

# This script contains the commands to generate most of the plots in the report
# using the 'plot_utils' utility in the msc_project module.
# Run 'python $PLOT_SCRIPT --help' for a list of the available subcommands (make sure PLOT_SCRIPT is defined).
# - The 'epochs' subcommand is used to plot Neurosim training vs. epochs, 'pytorch' is used for aihwkit.
# Run 'python $PLOT_SCRIPT <subcommand> --help' for more information on the available options for each subcommand.
#
# - requires a python environment with msc_project installed
# - remove the '--noshow' argument to show interactive plots

######################
### NEUROSIM PLOTS ###
######################
FILENAME="neurosim_accuracy_epochs.png"
python $PLOT_SCRIPT epochs "${RESULTS_DIR}/test33/run_1/neurosim" \
    -y accuracy --hue device_id \
    --xlabel "Epoch" --ylabel "Accuracy" --huelabel "Device ID" --title "NeuroSim MNIST Training" \
    --figsize 12 8 --legend_loc 'lower center' \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="neurosim_accuracy_numlevels.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test33/run_1.csv" \
    -x num_min_LTP_LTD -y accuracy --hue device_id \
    --xlabel "Number of conductance levels: min(LTP, LTD)" --ylabel "Accuracy" --huelabel "Device ID" --title "NeuroSim Accuracy vs. Min. Number of Conductance Levels" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="neurosim_accuracy_numlevels_artificial.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test34/run_1.csv" \
    -x val2 -y val1 --hue accuracy \
    --xlabel "Num. LTP Levels" --ylabel "Num. LTD Levels" --huelabel "Accuracy" --title "Accuracy vs. Num. Levels Map: Artificial Device" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="neurosim_pulses_per_synapse.png"
python $PLOT_SCRIPT epochs "${RESULTS_DIR}/test34/run_1_equal" \
    -y actual_pulse_number_per_synapse --hue val1 --notcumulative \
    --xlabel "Epoch" --ylabel "Average pulses applied per synapse" --huelabel "Number of levels (LTP=LTD)" --title "Neurosim Number of Pulses per Synapse" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="neurosim_accuracy_nonlinearity.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test33/run_1.csv" \
    -x NL_max -y accuracy --hue device_id \
    --xlabel "Nonlineality: max(LTP, LTD)" --ylabel "Accuracy" --huelabel "Device ID" --title "NeuroSim Accuracy vs. Max. Nonlinearity" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="neurosim_accuracy_nonlinearity_artificial.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test9/n30/" \
    -x nl -y accuracy \
    --xlabel "Nonlineality parameter" --ylabel "Accuracy" --title "Accuracy vs. Max. Nonlinearity: Artificial Device" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="neurosim_accuracy_pulsewidth.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test33/run_1.csv" \
    -x pulseWidth -y accuracy --hue device_id \
    --scale lin-log --filter device_id ID301XR1000Oct ID161ZR15000 ID170ZR5000 ID181ZR1000 --fmt "o-" \
    --xlabel "Pulse Width (s)" --ylabel "Accuracy" --huelabel "Device ID" --title "NeuroSim Accuracy vs. Pulse Width" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"


#####################
### AIHWKIT PLOTS ###
#####################
FILENAME="aihwkit_accuracy_dw.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test52/test52.csv" \
    -x granularity_up -y val_acc --hue device_id \
    --xlabel "Max ($\Delta w_0^u$, $\Delta w_0^d$)" --ylabel "Accuracy" --huelabel "Device ID" --title "AIHWKit Accuracy vs. $\Delta w_0$ (mixed-precision)" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

# ID161ZR15000
FILENAME="aihwkit_ID161ZR15000_accuracy_energy.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x energy -y val_acc --filter device_id ID161ZR15000 --hue pulseWidth --huescale log --scale lin-log \
    --xlabel "Energy (upper bound) [J]" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID161ZR15000 Accuracy vs. Energy" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID161ZR15000_accuracy_epoch.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x epoch -y val_acc --filter device_id ID161ZR15000 --hue pulseWidth --huescale log \
    --xlabel "Epoch" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID161ZR15000 Accuracy vs. Epoch" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID161ZR15000_accuracy_pulses.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x pulses -y val_acc --filter device_id ID161ZR15000 --hue pulseWidth --huescale log \
    --xlabel "Number of Applied Pulses" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID161ZR15000 Accuracy vs. Pulses" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

# ID181ZR1000
FILENAME="aihwkit_ID181ZR1000_accuracy_energy.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x energy -y val_acc --filter device_id ID181ZR1000 --hue pulseWidth --huescale log --scale lin-log \
    --exclude 'test_time' '12:37:23' '12:44:23' \
    --xlabel "Energy (upper bound) [J]" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID181ZR1000 Accuracy vs. Energy" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID181ZR1000_accuracy_epoch.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x epoch -y val_acc --filter device_id ID181ZR1000 --hue pulseWidth --huescale log \
    --exclude 'test_time' '12:37:23' '12:44:23' \
    --xlabel "Epoch" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID181ZR1000 Accuracy vs. Epoch" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID181ZR1000_accuracy_pulses.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x pulses -y val_acc --filter device_id ID181ZR1000 --hue pulseWidth --huescale log \
    --exclude 'test_time' '12:37:23' '12:44:23' \
    --xlabel "Number of Applied Pulses" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID181ZR1000 Accuracy vs. Pulses" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

# ID301XR1000Oct
FILENAME="aihwkit_ID301XR1000Oct_accuracy_energy.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x energy -y val_acc --filter device_id ID301XR1000Oct --hue pulseWidth --huescale log --scale lin-log \
    --exclude 'test_time' '15:35:56' '16:05:42' \
    --xlabel "Energy (upper bound) [J]" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID301XR1000Oct Accuracy vs. Energy" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID301XR1000Oct_accuracy_epoch.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x epoch -y val_acc --filter device_id ID301XR1000Oct --hue pulseWidth --huescale log \
    --exclude 'test_time' '15:35:56' '16:05:42' \
    --xlabel "Epoch" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID301XR1000Oct Accuracy vs. Epoch" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID301XR1000Oct_accuracy_pulses.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x pulses -y val_acc --filter device_id ID301XR1000Oct --hue pulseWidth --huescale log \
    --exclude 'test_time' '15:35:56' '16:05:42' \
    --xlabel "Number of Applied Pulses" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID301XR1000Oct Accuracy vs. Pulses" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

# ID171ZR5000
FILENAME="aihwkit_ID170ZR5000_accuracy_energy.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x energy -y val_acc --filter device_id ID170ZR5000 --hue pulseWidth --huescale log --scale lin-log \
    --xlabel "Energy (upper bound) [J]" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID170ZR5000 Accuracy vs. Energy" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID170ZR5000_accuracy_epoch.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x epoch -y val_acc --filter device_id ID170ZR5000 --hue pulseWidth --huescale log \
    --xlabel "Epoch" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID170ZR5000 Accuracy vs. Epoch" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

FILENAME="aihwkit_ID170ZR5000_accuracy_pulses.png"
python $PLOT_SCRIPT pytorch "${RESULTS_DIR}/test46/symmetric/aihwkit" \
    -x pulses -y val_acc --filter device_id ID170ZR5000 --hue pulseWidth --huescale log \
    --xlabel "Number of Applied Pulses" --ylabel "Accuracy" --huelabel "Pulse Width (s)" --title "ID170ZR5000 Accuracy vs. Pulses" \
    --shapes_sizes "${DATA_DIR}/LBE247_shapes_and_sizes.csv" --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

# onOffRatio vs pulseWidth
FILENAME="aihwkit_onOffRatio_pulseWidth.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test52/test52.csv" \
    -x pulseWidth -y onOffRatio --hue device_id --scale lin-log \
    --filter device_id ID301XR1000Oct ID161ZR15000 ID170ZR5000 ID181ZR1000 --fmt 'o-' \
    --xlabel "Pulse Width (s)" --ylabel "On/Off Ratio" --huelabel "Device ID" --title "AIHWKIT On/Off Ratio vs. Pulse Width" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow
echo "\e[32mGenerated figure:\e[0m ${FILENAME}"

# accuracy vs symmetry point
FILENAME="aihwkit_accuracy_sp.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test50/test50.csv" \
    -x symmetry_point -y val_acc --hue device_id \
    --xlabel "Symmetry Point" --ylabel "Accuracy" --huelabel "Device ID" --title "AIHWKIT Accuracy vs. Symmetry Point" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow

FILENAME="aihwkit_accuracy_sp_mp.png"
python $PLOT_SCRIPT summary "${RESULTS_DIR}/test52/test52.csv" \
    -x symmetry_point -y val_acc --hue device_id \
    --xlabel "Symmetry Point" --ylabel "Accuracy" --huelabel "Device ID" --title "AIHWKIT Accuracy vs. Symmetry Point (mixed-precision)" \
    --savefig "${FIGURES_DIR}/${FILENAME}" --noshow