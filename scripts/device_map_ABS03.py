import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
plt.style.use('ggplot')

from msc_project.utils import data_utils as du

DATA_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
START_ID = int(sys.argv[3])
PREFIX = f'StartTopLeftID{START_ID} _'
MODE = int(sys.argv[4]) # 0: Individual plots; 1: Combined plot, 2: Yield plot, 3: current range plot vs arc2 threshold
TITLE = sys.argv[5]
COL_NUM = 5
ROW_NUM = 4
STRIDE = -40 # by how much the device ID changes from one block to the next

COMPLIANCE_I = 1e-6
MIN_IMAX = 1e-9

V_READ = 0.1

ARC2_MIN = 200e-12
ARC2_10p = 1.6e-9
ARC2_1p = 16e-9

TOTAL_COLS = COL_NUM
TOTAL_ROWS = ROW_NUM * 4

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def get_coords(device_num):
    col = (device_num-1) % COL_NUM
    row = (device_num-1) // COL_NUM
    return col, row

subrow_dict = {0: 'W', 1: 'X', 2: 'Y', 3: 'Z'}
def get_id(col, row):
    base_id = (START_ID + col)  + STRIDE * (abs(row) // ROW_NUM)
    subrow = abs(row) % ROW_NUM
    return f"{base_id}.{subrow_dict[subrow]}R"

def get_minmax_I(df, V, get_abs=True):
    df = df[np.isclose(df['V1'], V, atol=0.001)]
    if get_abs:
        return abs(df['I1'].min()), abs(df['I1'].max())
    else:
        return df['I1'].min(), df['I1'].max()


# Get map files
csv_files = du.get_files(DATA_DIR, extension='.csv')

if MODE == 0:
    if not os.path.exists(OUT_DIR+'/device_plots'):
        os.makedirs(OUT_DIR+'/device_plots')
if MODE == 1 or MODE == 3:
    fig, axs = plt.subplots(TOTAL_ROWS, TOTAL_COLS, figsize=(2*TOTAL_COLS, 1.5*TOTAL_ROWS))
elif MODE == 2:
    idata = []

for f in csv_files:
    # Extract coordinates from the file name
    match = re.search(rf'{PREFIX} (\d+) (-?\d+)  \((\d+)', f)
    if match:
        x_coord = int(match.group(1))
        y_coord = int(match.group(2))
        device_num = int(match.group(3))
        col, row = get_coords(device_num)
        device_id = get_id(col, row)
        print(f"Coordinates: x={x_coord}, y={y_coord}, num={device_num}")
        print(f"\t Col={col}, Row={row}, Device ID: {device_id}")
        print()

        # Load the data
        df = pd.read_csv(f, skiprows=258, usecols=[' V1', ' I1', ' R', ' Iabs'])
        df.columns = df.columns.str.strip() # remove annoying whitespaces from column names
        
        # Plot the data
        if MODE == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
        elif MODE == 1:
            ax = axs[row, col]
        elif MODE == 2:
            idata.append([col, row, df['Iabs'].max()])
        elif MODE == 3:
            ax = axs[row, col]
            if (df['I1'].max() >= COMPLIANCE_I):
                ax.set_facecolor('black')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.grid(False)
                continue
            imin_n, imax_n = get_minmax_I(df, -V_READ)
            imin_p, imax_p = get_minmax_I(df, V_READ)
            # print(f"I range: {imin_n:.2e} - {imax_n:.2e} (V={-V_READ})")
            # print(f"I range: {imin_p:.2e} - {imax_p:.2e} (V={V_READ})")
            print()
            ax.barh(-V_READ, width=imax_n-imin_n, height=0.1, left=imin_n, color='blue')
            ax.barh(V_READ, width=imax_p-imin_p, height=0.1, left=imin_p, color='blue')
            # arc2 current ranges: orange (min, 10%), yellow (10%, 1%), green (1% up)
            ax.barh(0, width=ARC2_10p-ARC2_MIN, height=0.05, left=ARC2_MIN, color='orange')
            ax.barh(0, width=ARC2_1p-ARC2_10p, height=0.05, left=ARC2_10p, color='yellow')
            ax.barh(0, width=COMPLIANCE_I-ARC2_1p, height=0.05, left=ARC2_1p, color='green')
            ax.set(xscale='linear', xlim=(min(imin_n, imin_p), max(imax_n, imax_p)), title=f"{device_id} ({device_num})")

            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        if MODE == 0 or MODE == 1:
            ax2 = ax.twinx()
            ax.plot(df['V1'], df['I1'], color='blue', label='I')

            # Remap df['Iabs'] to the same range as df['I1']
            df['Iabs'] = (df['Iabs'] - df['Iabs'].min()) / (df['Iabs'].max() - df['Iabs'].min()) * (df['I1'].max() - df['I1'].min()) + df['I1'].min()
            ax.plot(df['V1'], df['Iabs'], color='green', label='Iabs')
            ax2.plot(df['V1'], df['R'], color='orange', label='R')

        if MODE == 0:
            ax.set_ylabel('I (A)', color='blue')
            ax2.set_ylabel(r'R ($\Omega$)', color='orange')

            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='orange')

            ax.set(xlabel='Voltage (V)', title=f"{device_id} (row={row}, col={col}, num={device_num})")
            ax.legend()
            ax2.legend()

            plt.savefig(f'{OUT_DIR}/device_plots/map_ID{device_id}_row{row}_col{col}_n{device_num}.png')
        elif MODE == 1:
            ax.set(title=f"{device_id} ({device_num})")
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
        row.set_ylabel(i, rotation=0, size='large')
    fig.suptitle(TITLE)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.savefig(f'{OUT_DIR}/fullmap.png')
    plt.show()

if MODE == 2:
    idata = np.array(idata)
    shortdata = idata[idata[:,2] >= COMPLIANCE_I]
    fig, ax = plt.subplots(figsize=(1.5*TOTAL_COLS, TOTAL_ROWS))
    im = ax.scatter(idata[:,0], -idata[:,1], c=np.log10(idata[:,2]), cmap='plasma', s=100)
    ax.scatter(shortdata[:,0], -shortdata[:,1], c='black', marker='x', s=100, label='Shorted')
    ax.hlines([-3.5, -7.5, -11.5], xmin=-0.5, xmax=COL_NUM-0.5, color='black', lw=1)
    fig.colorbar(im, label='$log_{10}$(Max abs. current)')
    im.set_clim(vmin=np.log10(MIN_IMAX), vmax=np.log10(COMPLIANCE_I))
    fig.suptitle(TITLE)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/yield_map.png')
    plt.show()

if MODE == 3:
    for i,col in enumerate(axs[-1]):
        col.set_xlabel(i, rotation=0, size='large')
    for i,row in enumerate(axs[:,0]):
        row.set_ylabel(i, rotation=0, size='large')
    fig.suptitle(TITLE)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.savefig(f'{OUT_DIR}/current_range.png')
    plt.show()