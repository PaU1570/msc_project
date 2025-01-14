#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

DATA_DIR="/scratch/msc24h18/msc_project/data/LBE247_analyzed_2nd_run_only"
RESULTS_DIR="/scratch/msc24h18/msc_project/results"

# run test31
#bash "${SCRIPT_DIR}/test31.sh" "${DATA_DIR}" "${RESULTS_DIR}/test31"

# run test32a
#bash "${PARENT_DIR}/aihwkit/run_mnist_mp.sh" "${DATA_DIR}" "${RESULTS_DIR}/test32/no_out_scaling_wnstd_0" 25 0.3 0.3 0 0.3 0.3
# run test32b
#bash "${PARENT_DIR}/aihwkit/run_mnist_mp_learnoutscaling.sh" "${DATA_DIR}" "${RESULTS_DIR}/test32/learn_out_scaling_wnstd_0" 25 0.3 0.3 0 0.3 0.3
# run test32c
#bash "${PARENT_DIR}/aihwkit/run_mnist_mp.sh" "${DATA_DIR}" "${RESULTS_DIR}/test32/no_out_scaling_wnstd_0.3" 25 0.3 0.3 0.3 0.3 0.3
# run test32d
#bash "${PARENT_DIR}/aihwkit/run_mnist_mp_learnoutscaling.sh" "${DATA_DIR}" "${RESULTS_DIR}/test32/learn_out_scaling_wnstd_0.3" 25 0.3 0.3 0.3 0.3 0.3

# run test33
#bash "${PARENT_DIR}/run_from_configs.sh" "${RESULTS_DIR}/test33" "${RESULTS_DIR}/test33" 3

# run test34
#bash "${PARENT_DIR}/run_from_configs.sh" "/scratch/msc24h18/msc_project/neurosim_configs/test10" "${RESULTS_DIR}/test34" 5

# run test35
bash "${SCRIPT_DIR}/test35.sh" "${DATA_DIR}" "${RESULTS_DIR}/test35"

# run test36
bash "${SCRIPT_DIR}/test36.sh" "${DATA_DIR}" "${RESULTS_DIR}/test36"

# run test37a
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_none.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_none_no_out_scaling_wnstd_0" 25 0.3 0.3 0 0.3 0.3
# run test37b
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_none_learnoutscaling.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_none_learn_out_scaling_wnstd_0" 25 0.3 0.3 0 0.3 0.3
# run test37c
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_none.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_none_no_out_scaling_wnstd_0.3" 25 0.3 0.3 0.3 0.3 0.3
# run test37d
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_none_learnoutscaling.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_none_learn_out_scaling_wnstd_0.3" 25 0.3 0.3 0.3 0.3 0.3
# run test37e
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_stochastic.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_stochastic_no_out_scaling_wnstd_0" 25 0.3 0.3 0 0.3 0.3
# run test37f
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_stochastic_learnoutscaling.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_stochastic_learn_out_scaling_wnstd_0" 25 0.3 0.3 0 0.3 0.3
# run test37g
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_stochastic.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_stochastic_no_out_scaling_wnstd_0.3" 25 0.3 0.3 0.3 0.3 0.3
# run test37h
bash "${PARENT_DIR}/aihwkit/run_mnist_mp_pt_stochastic_learnoutscaling.sh" "${DATA_DIR}" "${RESULTS_DIR}/test37/pt_stochastic_learn_out_scaling_wnstd_0.3" 25 0.3 0.3 0.3 0.3 0.3

