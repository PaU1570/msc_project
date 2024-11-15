import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from msc_project.utils import data_utils as du

DATA_DIR = 'data/20241114_BeFerro_FTJ10'
PREFIX = 'BeFerroFTJ10_Map3_ _'
OUT_DIR = 'results/20241114_BeFerro_FTJ10_map'
MODE = 0 # 0: Individual plots; 1: Combined plot
# Get map files
csv_files = du.get_files(DATA_DIR, extension='.csv', contains='Map3')

if MODE == 1:
    fig, axs = plt.subplots(13, 30, figsize=(30, 13))

for f in csv_files:
    # Extract coordinates from the file name
    match = re.search(rf'{PREFIX} (\d+) (\d+)', f)
    if match:
        x_coord = int(match.group(1))
        y_coord = int(match.group(2))
        print(f"Coordinates: x={x_coord}, y={y_coord}")

        # Load the data
        df = pd.read_csv(f, skiprows=258, usecols=[' V1', ' I1', ' ROh', ' Iabs'])
        df.columns = df.columns.str.strip() # remove annoying whitespaces from column names
        
        # Plot the data
        if MODE == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = axs[12 - y_coord, x_coord]
        ax2 = ax.twinx()
        ax.plot(df['V1'], df['I1'], color='blue', label='I')

        # Remap df['Iabs'] to the same range as df['I1']
        df['Iabs'] = (df['Iabs'] - df['Iabs'].min()) / (df['Iabs'].max() - df['Iabs'].min()) * (df['I1'].max() - df['I1'].min()) + df['I1'].min()
        ax.plot(df['V1'], df['Iabs'], color='green', label='Iabs')
        ax2.plot(df['V1'], df['ROh'], color='orange', label='ROh')

        if MODE == 0:
            ax.set_ylabel('I (A)', color='blue')
            ax2.set_ylabel(r'R ($\Omega$)', color='orange')

            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='orange')

            ax.set(xlabel='Voltage (V)', title=f"Map3: x={x_coord}, y={y_coord}")
            ax.legend()
            ax2.legend()

            plt.savefig(f'{OUT_DIR}/device_plots/map_x{x_coord}_y{y_coord}.png')
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
    else:
        print(f"No coordinates found in the file name: {os.path.basename(f)}")

if MODE == 1:
    for i,col in enumerate(axs[-1]):
        col.set_xlabel(i, rotation=0, size='large')
    for i,row in enumerate(axs[:,0]):
        row.set_ylabel(12 - i, rotation=0, size='large')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fullmap.png')