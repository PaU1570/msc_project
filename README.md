# Overview
This repository contains the code for my master's thesis for MSc Quantum Engineering at ETHZ, titled "Ferroelectric Crossbars for Artificial Neural Networks"

# Data
The raw data is in ```data/LBE247``` and ```data/20241024_LBE247```.
The processed data to be fed into the simulations is in ```data/LBE247_analyzed_2nd_run_only```. This data only contains the second (or, in general, last) run of each measurement instance (indicated by the numbers in parethesis on each csv file in the raw data). This processed data can be generating by running the following commands:
```
python src/msc_project/utils/analyze_utils.py data/LBE247 <processed_data_path> --cutoffs data/LBE247_cutoffs.csv

python src/msc_project/utils/analyze_utils.py data/20241024_LBE247 <processed_data_path> --cutoffs data/20241024_LBE247_cutoffs.csv
```
Replace ```<processed_data_path>``` with the location where you want to store the processed data. The cutoffs file is optional; it trims the data according to the cutoffs specified in the file. If given, measurements that do not appear in the file will be ignored. There is an interactive flask app to visually generate the cutoffs file:
```
python apps/interactive-cutoffs/app.py /path/to/raw/data /output/file.csv
```

# Simulations
The outputs of the simulations are stored in ```results/```. The file ```results/description.txt``` contains a short description of each run.

## Running NeuroSim simulations
Make sure the ```MLP_NeuroSim_V3.0``` submodule is initialized and you have compiled NeuroSim. See the NeuroSim manual for instructions. This fork of NeuroSim has been modified to accept json configuration files without recompiling every time.

To run NeuroSim from already available config files:
```
./scripts/run_from_configs.sh <configs directory> <output directory> <number of runs>
```
To run from raw data:
```
./scripts/run_all.sh <data directory> <output directory> <reference configuration>
```

## Running AIHWKIT simulations
There are different scripts available in ```scripts/aihwkit/```. For example, to run mnist with mixed-precision:
```
./scripts/aihwkit/run_mnist_mp.sh <analyzed data directory> <output directory> [epochs (25)] [dw_min_dtod (0.3)] [dw_min_std (0.3)] [write_noise_std (0.0)] [w_min_dtod (0.3)] [w_max_dtod (0.3)]
```
Some options (e.g. pulse counting, asymmetric pulsing) require the custom version of aihwkit. Follow the development setup instructions form their documentation and make sure you are using the custom version. Other scripts work fine with the default version installed from conda/pypi.

## Plotting
The script ```src/msc_project/utils/plot_utils.py``` is able to generate a wide range of plots based on the simulation results. The commands to generate most plots in the report can be found in ```results/report_figures/generate/generate_figures.sh```.