import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import LogNorm
import seaborn as sns
from plot_utils import create_parser
import os
from run_neurosim_to_csv import get_data_from_file

plt.style.use('ggplot')

if __name__ == "__main__":
    
    args = create_parser()

    files = os.listdir(args.input)
    files = [f for f in files if f.endswith('.dat')]

    title = args.title if args.title is not None else f'{args.input}'

    data = dict()
    epochs = []
    accuracy = []
    read_latency = []
    write_latency = []
    read_energy = []
    write_energy = []
    for f in files:
        tmp, tmp_epoch_num, tmp_accuracy, tmp_rl, tmp_wl, tmp_re, tmp_we = get_data_from_file(os.path.join(args.input, f), energy=True)
        if not data:
            data = {key: [value] for key, value in tmp.items()}
        else:
            for key, value in tmp.items():
                data[key].append(value)
        epochs.append(tmp_epoch_num)
        accuracy.append(tmp_accuracy)
        read_latency.append(tmp_rl)
        write_latency.append(tmp_wl)
        read_energy.append(tmp_re)
        write_energy.append(tmp_we)

    # dataframe order matches the order of epochs and accuracy (i.e. epoch[i], accuracy[i] correspond to the i-th row of data)
    data = pd.DataFrame(data)
    data = data.apply(pd.to_numeric, errors='ignore')   
    epochs = np.array(epochs)
    # select the y-axis value
    if args.y == 'accuracy':
        yvals = np.array(accuracy)
    elif args.y == 'read_latency':
        yvals = np.array(read_latency)
    elif args.y == 'write_latency':
        yvals = np.array(write_latency)
    elif args.y == 'read_energy':
        yvals = np.array(read_energy)
    elif args.y == 'write_energy':
        yvals = np.array(write_energy)

    if args.filter is not None:
        if args.filter[0] not in data.columns:
            print(f"Error: {args.filter[0]} is not a column in the data.")
            exit(1)
        print(f'Filtering data based on {args.filter[0]} = {args.filter[1]}')
        try:
            filter_value = pd.to_numeric(args.filter[1])
        except ValueError:
            filter_value = args.filter[1]

        data = data[data[args.filter[0]] == filter_value]
        if data.empty:
            print(f"Error: No data found for {args.filter[0]} = {args.filter[1]}.")
            exit(1)

    fig, ax = plt.subplots(figsize=(8, 6))

    if data[args.hue].dtype == 'float64':
        cmap = colormaps.get_cmap('plasma')
        if args.huescale == 'log':
            norm = LogNorm(data[args.hue].min(), data[args.hue].max())
        else:
            norm = plt.Normalize(data[args.hue].min(), data[args.hue].max())
        for i in data.index:
            color = cmap(norm(data[args.hue][i]))
            ax.plot(epochs[i], yvals[i], color=color)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, label=args.hue, ax=ax)

    else:
        unique_types = data[args.hue].unique()
        #colors = sns.color_palette('husl', len(unique_types))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
        for unique_type, color in zip(unique_types, colors):
            type_indices = data[data[args.hue] == unique_type].index
            for i in type_indices:
                ax.plot(epochs[i], yvals[i], label=unique_type, color=color)
        # make sure repeated labels are only shown once
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title=args.hue)

    ax.set(xlabel='Epoch', ylabel=args.y, title=title, aspect='auto')

    if args.savefig is not None:
        plt.tight_layout()
        plt.savefig(args.savefig)
    plt.show()

        
