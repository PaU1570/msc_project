import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
from matplotlib import colormaps
import matplotlib.animation as animation
import seaborn as sns

import sys
from msc_project.utils.run_neurosim_to_csv import get_data_from_file
import msc_project.utils.data_utils as du

def create_parser():
    """
    Create a parser for the command line arguments. Different subcommands are available.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    common_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Select sub-command', dest='subcommand')

    # input argument
    common_parser.add_argument("input", type=str, help="Path to the input file or directory")
    # plot appearance arguments
    common_parser.add_argument('--title', type=str, help='Title of the plot', default=None)
    common_parser.add_argument('--xlabel', type=str, help='X axis label', default=None)
    common_parser.add_argument('--ylabel', type=str, help='Y axis label', default=None)
    common_parser.add_argument('--huelabel', type=str, help='Hue label', default=None)
    common_parser.add_argument('--xlim', type=float, nargs=2, help='X axis limits', default=None)
    common_parser.add_argument('--ylim', type=float, nargs=2, help='Y axis limits', default=None)
    common_parser.add_argument('--scale', type=str, choices=['linear', 'log-log', 'log-lin', 'lin-log'], default='linear', help='Scale of the axes')
    common_parser.add_argument('--aspect', type=str, help='Aspect ratio of the plot', default='auto')
    # data selection arguments
    common_parser.add_argument('--filter', type=str, help='Filter the data based on a column (multiple filters can be specified)', default=None, action='append', nargs='*')
    common_parser.add_argument('--exclude', type=str, help='Exclude the data based on a column (multiple filters can be specified)', default=None, action='append', nargs='*')
    common_parser.add_argument('--hue', type=str, help='Which attribute to use for coloring the data points', default=None)
    common_parser.add_argument('--huescale', type=str, choices=['linear', 'log'], default='linear', help='Scale of the hue')
    # save figure argument
    common_parser.add_argument('--savefig', type=str, help='Save the figure to a file', default=None)

    # subparser arguments
    parser_summary = subparsers.add_parser('summary', help='Plot any combination of columns from a csv summary file', parents=[common_parser])
    parser_summary.add_argument('-y', type=str, help='Y axis column')
    parser_summary.add_argument('-x', type=str, help='X axis column')
    parser_summary.add_argument('--all', action='store_true', help='Plot all pairs of columns in a seaborn pairplot')
    parser_summary.set_defaults(func=plot_summary)

    parser_epochs = subparsers.add_parser('epochs', help='Plot training evolution over epochs', parents=[common_parser])
    parser_epochs.add_argument('-y', type=str, help='Y axis value', choices=['accuracy',
                                                                             'read_latency',
                                                                             'write_latency',
                                                                             'read_energy',
                                                                             'write_energy',
                                                                             'weight_update',
                                                                             'pulse_number',
                                                                             'actual_conductance_update',
                                                                             'actual_pulse_number',
                                                                             'actual_conductance_update_per_synapse',
                                                                             'actual_pulse_number_per_synapse'], default='accuracy', nargs='*')
    parser_epochs.add_argument('--notcumulative', action='store_true', help='Plot only the change per epoch, not the cumulative value')
    parser_epochs.add_argument('--norm', action='store_true', help='Normalize the y-axis values')
    parser_epochs.set_defaults(func=plot_epochs)

    parser_pytorch = subparsers.add_parser('pytorch', help='Plot training evolution from a pytorch log file', parents=[common_parser])
    parser_pytorch.add_argument('-x', type=str, help='X axis value', choices=['epoch',
                                                                              'step',
                                                                              'test_acc',
                                                                              'test_loss',
                                                                              'train_loss',
                                                                              'val_acc',
                                                                              'val_loss'], default='epoch')
    parser_pytorch.add_argument('-y', type=str, help='Y axis value', choices=['epoch',
                                                                              'step',
                                                                              'test_acc',
                                                                              'test_loss',
                                                                              'train_loss',
                                                                              'val_acc',
                                                                              'val_loss'], default='val_acc', nargs='*')
    parser_pytorch.add_argument('--smooth', type=int, help='Number of points to use for smoothing')
    parser_pytorch.set_defaults(func=plot_pytorch)

    args = parser.parse_args()

    # check summary args
    if args.subcommand == 'summary':
        if (args.x or args.y) and args.all:
            parser.error("Cannot specify -x and -y when using --all")
        if (args.x and not args.y) or (args.y and not args.x):
            parser.error("Must specify both -x and -y")
        if not (args.x or args.y) and not args.all:
            parser.error("Must specify either -x and -y or --all")
    elif args.subcommand == 'epochs' or args.subcommand == 'pytorch':
        pass # no checks yet
    else:
        parser.error("Please specify a subcommand")

    return args

def filter_data(data, filter_list, exclusion_list, reset_index=True):
    """
    Filter the data based on a column and a value.

    Args:
        data (pd.DataFrame): The data to filter
        filter_list (list): A list of filters, each containing a column name and a value. I.e. [['column1', 'value1'], ['column2', 'value2'], ...]
        exclusion_list (list): A list of exclusions, each containing a column name and a value. I.e. [['column1', 'value1'], ['column2', 'value2'], ...].
        
    Returns:
        pd.DataFrame: The filtered data
    """

    if filter_list is not None:
        for filter_column, filter_value in filter_list:
            if filter_column not in data.columns and filter_column != 'fit_R2':
                print(f"Error: {filter_column} is not a column in the data.")
                exit(1)
            
            comp = '='
            if filter_column == 'fit_R2_LTD' or filter_column == 'fit_R2_LTP' or filter_column == 'fit_R2':
                comp = '>'

            print(f'Filtering data based on {filter_column} {comp} {filter_value}')
            try:
                filter_value = pd.to_numeric(filter_value)
            except ValueError:
                pass
            if comp == '=':
                data = data[data[filter_column] == filter_value]
            elif comp == '>':
                if filter_column == 'fit_R2':
                    data = data[(data['fit_R2_LTD'] > filter_value) & (data['fit_R2_LTP'] > filter_value)]
                else:
                    data = data[data[filter_column] > filter_value]
            if data.empty:
                print(f"Error: No data found for {filter_column} {comp} {filter_value}.")
                exit(1)

    if exclusion_list is not None:
        for exclusion_column, exclusion_value in exclusion_list:
            if exclusion_column not in data.columns:
                print(f"Warning: {exclusion_column} is not a column in the data. No data will be excluded based on this column.")
            print(f'Excluding data based on {exclusion_column} = {exclusion_value}')
            data = data[data[exclusion_column] != exclusion_value]
    
    if reset_index:
        data = data.reset_index(drop=True)
    return data

def read_average_data(path):
    """
    Read all the csv files in a directory into a dataframe with average values and standard deviations.

    Args:
        path (str): The path to the directory containing the csv files. All files are expected to have the same rows and columns.

    Returns:
        pd.DataFrame: The dataframe with average values and standard deviations
    """
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.csv')]
    data = None
    for f in files:
        if data is None:
            data = pd.read_csv(os.path.join(path, f))
        else:
            data = pd.concat([data, pd.read_csv(os.path.join(path, f))])

    if ('device_id' in data.columns) and ('test_date' in data.columns) and ('test_time' in data.columns):
        groupby_cols = ['device_id', 'test_date', 'test_time']
    elif 'filename' in data.columns:
        groupby_cols = ['filename']
    else:
        print("Error: Can't match data between files. The data does not contain columns ['device_id', 'test_date', 'test_time'] or ['filename'].")
        exit(1)

    data['avgAccuracy'] = data.groupby(groupby_cols)['accuracy'].transform('mean')
    data['stdAccuracy'] = data.groupby(groupby_cols)['accuracy'].transform('std')
    data.drop_duplicates(subset=groupby_cols, inplace=True)
    data.drop(columns=['accuracy'], inplace=True)
    data.rename(columns={'avgAccuracy': 'accuracy'}, inplace=True)
    return data

def set_axis_properties(ax, args):
    """
    Set the properties of the axis based on the arguments.

    Args:
        ax (matplotlib.axes.Axes): The axis to set the properties for
        args (argparse.Namespace): The arguments passed to the function

    Returns:
        None
    """
    if type(ax) is not plt.Axes:
        print("Error: ax argument to set_axis_properties() must be a matplotlib Axes object.")
        exit(1)
    
    xscale = 'linear' if args.scale in ['linear', 'log-lin'] else 'log'
    yscale = 'linear' if args.scale in ['linear', 'lin-log'] else 'log'
    title = args.title if args.title is not None else f'{args.input}'
    xlabel = args.xlabel if args.xlabel is not None else args.x
    ylabel = args.ylabel if args.ylabel is not None else args.y
    if type(ylabel) is list:
        ylabel = ', '.join(ylabel)

    ax.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale, title=title, aspect=args.aspect, xlim=args.xlim, ylim=args.ylim)

def plot_summary(args):

    isFile = os.path.isfile(args.input)
    isDir = os.path.isdir(args.input)
    if not isFile and not isDir:
        print(f"Error: The file or directory {args.input} does not exist.")
        exit(1)

    if isFile:
        data = pd.read_csv(args.input)
    elif isDir:
        data = read_average_data(args.input)

    data = filter_data(data, args.filter, args.exclude)
    data = data.sort_values(by=['device_id', 'test_time'])

    if not args.all:
        if args.x not in data.columns :
            print(f"Error: {args.x} is not a column in the data.")
            exit(1)
        if args.y not in data.columns :
            print(f"Error: {args.y} is not a column in the data.")
            exit(1)

        xerr = data['stdAccuracy'].to_list() if args.x == 'accuracy' and isDir else None
        yerr = data['stdAccuracy'].to_list() if args.y == 'accuracy' and isDir else None

        fig, ax = plt.subplots(figsize=(8, 6))

        huelabel = args.huelabel if args.huelabel is not None else args.hue
        if args.hue is not None and data[args.hue].dtype == 'float64':
            norm = clr.LogNorm() if args.huescale == 'log' else clr.Normalize()       
            cmap = plt.cm.plasma
            colors = cmap(norm(data[args.hue].values))

            cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
            cbar.set_label(huelabel)

            if xerr is None:
                xerr = np.zeros(len(data[args.x]))
            if yerr is None:
                yerr = np.zeros(len(data[args.y]))

            for x, y, ex, ey, color in zip(data[args.x], data[args.y], xerr, yerr, colors):
                ax.errorbar(x, y, xerr=ex, yerr=ey, fmt='o', capsize=2, color=color, ecolor=color)

        else:
            unique_types = data[args.hue].unique() if args.hue is not None else ['data']
            #colors = sns.color_palette('husl', len(unique_types))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
            for unique_type, color in zip(unique_types, colors):
                type_data = data[data[args.hue] == unique_type] if args.hue is not None else data
                ax.errorbar(type_data[args.x], type_data[args.y], xerr=xerr, yerr=yerr, fmt='o', capsize=2, color=color, ecolor=color, label=unique_type)
            ax.legend(title=huelabel)

        set_axis_properties(ax, args)

        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig, facecolor=fig.get_facecolor())
        plt.show()

    if args.all:
        cols = ['stepSize','pulseWidth','onOffRatio','accuracy', 'A_LTP', 'A_LTD']
        g = sns.pairplot(data, hue=args.hue, vars=cols)
        g.figure.suptitle(args.title if args.title is not None else f'{args.input}')
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig)
        plt.show()

def plot_epochs(args):

    mode = 'dir'
    if os.path.isdir(args.input):
        files = os.listdir(args.input)
        files = [f for f in files if f.endswith('.dat')]
    elif os.path.isfile(args.input):
        files = [args.input]
        mode = 'file'
    else:
        print(f"Error: The file or directory {args.input} does not exist.")
        exit(1)

    data = dict()
    epoch_data = {"epochs": [],
                  "accuracy": [],
                  "read_latency": [],
                  "write_latency": [],
                  "read_energy": [],
                  "write_energy": [],
                  "weight_update": [],
                  "pulse_number": [],
                  "actual_conductance_update": [],
                  "actual_pulse_number": [],
                  "actual_conductance_update_per_synapse": [],
                  "actual_pulse_number_per_synapse": []}
    
    for f in files:
        filename = os.path.join(args.input, f) if mode == 'dir' else args.input
        tmp, tmp_epoch_num, tmp_accuracy, tmp_rl, tmp_wl, tmp_re, tmp_we, tmp_wu, tmp_pn, tmp_acu, tmp_apn, tmp_acups, tmp_apnps = get_data_from_file(filename)
        if not data:
            data = {key: [value] for key, value in tmp.items()}
        else:
            for key, value in tmp.items():
                data[key].append(value)
        epoch_data['epochs'].append(tmp_epoch_num)
        epoch_data['accuracy'].append(tmp_accuracy)
        epoch_data['read_latency'].append(tmp_rl)
        epoch_data['write_latency'].append(tmp_wl)
        epoch_data['read_energy'].append(tmp_re)
        epoch_data['write_energy'].append(tmp_we)
        epoch_data['weight_update'].append(tmp_wu)
        epoch_data['pulse_number'].append(tmp_pn)
        epoch_data['actual_conductance_update'].append(tmp_acu)
        epoch_data['actual_pulse_number'].append(tmp_apn)
        epoch_data['actual_conductance_update_per_synapse'].append(tmp_acups)
        epoch_data['actual_pulse_number_per_synapse'].append(tmp_apnps)

    # dataframe order matches the order of epochs and accuracy (i.e. epoch_data['epochs'][i], epochs_data['accuracy'][i] correspond to the i-th row of data)
    data = pd.DataFrame(data)
    data = data.apply(pd.to_numeric, errors='ignore')   
    epochs = np.array(epoch_data['epochs'])
    # select the y-axis value
    if args.y == 'accuracy':
        args.y = ['accuracy']
    if mode == 'dir' and len(args.y) > 1:
        print("Error: Only one y-axis value can be selected when using a directory as input.")
        exit(1)
    yvals = np.array([epoch_data[a] for a in args.y])
    epochs = np.squeeze(epochs)
    yvals = np.squeeze(yvals, axis=0 if mode == 'dir' else 1)
    if args.notcumulative:
        yvals = np.diff(yvals, prepend=0, axis=1)
    if args.norm:
        max = np.max(yvals, axis=1)[:, np.newaxis]
        min = np.min(yvals, axis=1)[:, np.newaxis]
        yvals = (yvals - min) / (max - min)

    data = filter_data(data, args.filter, args.exclude, reset_index=False)
    data = data.sort_values(by=['device_id', 'test_time'])

    fig, ax = plt.subplots(figsize=(8, 6))

    huelabel = args.huelabel if args.huelabel is not None else args.hue
    if args.hue is not None and data[args.hue].dtype == 'float64' and mode == 'dir':
        cmap = colormaps.get_cmap('plasma')
        if args.huescale == 'log':
            norm = clr.LogNorm(data[args.hue].min(), data[args.hue].max())
        else:
            norm = clr.Normalize(data[args.hue].min(), data[args.hue].max())
        for i in data.index:
            color = cmap(norm(data[args.hue][i]))
            ax.plot(epochs[i], yvals[i], color=color)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, label=huelabel, ax=ax)

    elif mode == 'dir':
        unique_types = data[args.hue].unique() if args.hue is not None else ['data']
        #colors = sns.color_palette('husl', len(unique_types))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
        for unique_type, color in zip(unique_types, colors):
            type_indices = data[data[args.hue] == unique_type].index if args.hue is not None else data.index
            for i in type_indices:
                ax.plot(epochs[i], yvals[i], label=unique_type, color=color)
        # make sure repeated labels are only shown once
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title=huelabel)

    else:
        labels = args.y
        for i in range(len(yvals)):
            ax.plot(epochs, yvals[i], label=labels[i])
        ax.legend(title=huelabel)
        
    args.xlabel = 'Epochs'
    set_axis_properties(ax, args)

    if args.savefig is not None:
        plt.tight_layout()
        plt.savefig(args.savefig)
    plt.show()

def plot_pytorch(args):
    mode = 'dir'
    if os.path.isdir(args.input):
        files = du.get_metrics_csv_files(args.input)
    elif os.path.isfile(args.input):
        files = [args.input]
        mode = 'file'
    else:
        print(f"Error: The file or directory {args.input} does not exist.")
        exit(1)

    if type(args.y) is not list:
        args.y = [args.y]
    if mode == 'dir' and len(args.y) > 1:
        print("Error: Only one y-axis value can be selected when using a directory as input.")
        exit(1)

    metrics = [pd.read_csv(f) for f in files]
    metrics = [m[m[args.x].notna()] for m in metrics]

    # get metadata from RPU_Config.txt files
    rpu_config_files = du.get_rpu_txt_files(args.input)
    rpu_configs = [du.read_rpu_txt(rpu) for rpu in rpu_config_files]
    rpu_configs = {key: np.array([d[key] for d in rpu_configs]) for key in rpu_configs[0]}

    # get metadata from Summary.dat files
    summary_files = du.get_summary_files(args.input)
    summaries = [du.read_summary_file(s)[0] for s in summary_files]

    try:
        summaries = {key: [d[key] for d in summaries] for key in summaries[0].keys()}
    except KeyError: # due to a mistake some files in test16 are older than others and have different keys, so use only a few keys that all files have
        keys = ['device_name', 'device_id', 'test_date', 'test_time', 'stepSize', 'pulseWidth', 'onOffRatio']
        print("Using only the following keys from the Summary.dat files:", keys)
        summaries = {key: [d[key] for d in summaries] for key in keys}
        

    metadata = {**summaries, **rpu_configs}
    metadata = pd.DataFrame(metadata)
    metadata = metadata.apply(pd.to_numeric, errors='ignore')
    metadata = filter_data(metadata, args.filter, args.exclude, reset_index=False)
    metadata = metadata.sort_values(by=['device_id', 'test_time'])

    if args.hue is not None and args.hue not in metadata.keys():
        print(f"Error: {args.hue} is not a valid hue. Options are {list(metadata.keys())}")
        exit(1)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    huelabel = args.huelabel if args.huelabel is not None else args.hue
    if mode == 'file':
        colors = plt.cm.rainbow(np.linspace(0, 1, len(args.y)))
        for metric in metrics:
            for i,y in enumerate(args.y):
                m = metric[(metric[y].notna()) & (metric[args.x].notna())]
                m = m.groupby(args.x).mean().reset_index()
                ax.plot(m[args.x], m[y], color=colors[i], alpha=0.5 if args.smooth else 1, label=y)
                if args.smooth is not None:
                    smoothed_metrics = m.rolling(window=args.smooth).mean()
                    ax.plot(smoothed_metrics[args.x], smoothed_metrics[y], color=colors[i])
        ax.legend()
    else:
        hue_type = 'none' if args.hue is None else 'numeric' if metadata[args.hue].dtype == 'float64' else 'categorical'
        if hue_type == 'none':
            colors = plt.cm.rainbow(np.linspace(0, 1, len(metrics)))
        elif hue_type == 'numeric':
            cmap = colormaps.get_cmap('plasma')
            if args.huescale == 'log':
                norm = clr.LogNorm(metadata[args.hue].min(), metadata[args.hue].max())
            else:
                norm = clr.Normalize(metadata[args.hue].min(), metadata[args.hue].max())
        else:
            unique_types = metadata[args.hue].unique()
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
            color_dict = {unique_type: color for unique_type, color in zip(unique_types, colors)}

        for i in metadata.index:
            try:
                metric = metrics[i]
                m = metric[(metric[args.y[0]].notna()) & (metric[args.x].notna())]
                m = m.groupby(args.x).mean().reset_index()

                if hue_type == 'none':
                    color = colors[i]
                elif hue_type == 'numeric':
                    color = cmap(norm(metadata[args.hue][i]))
                else:
                    color = color_dict[metadata[args.hue][i]]

                alpha = 0.5 if args.smooth else 1
                label = metadata[args.hue][i] if hue_type == 'categorical' else None
                ax.plot(m[args.x], m[args.y[0]], color=color, alpha=alpha, label=label)
                if args.smooth is not None:
                    smoothed_metrics = m.rolling(window=args.smooth).mean()
                    ax.plot(smoothed_metrics[args.x], smoothed_metrics[args.y[0]], color=color)
            except IndexError:
                print(f"Error: No data found for index {i}")
                pass

        if hue_type == 'numeric':
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, label=huelabel, ax=ax)
        elif hue_type == 'categorical':
            # make sure repeated labels are only shown once
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title=huelabel)

    set_axis_properties(ax, args)

    if args.savefig is not None:
        plt.tight_layout()
        plt.savefig(args.savefig)
    plt.show()


######################################################
# Functions for plotting weights #####################
######################################################

def _plot_weight_hist(ax, w, layers=(True, True, True), label=None, color=None):
    l1 = w['0.analog_module'][0]
    l2 = w['2.analog_module'][0]
    l3 = w['4.analog_module'][0]

    bins = np.linspace(-1.5, 1.5, 100)

    if layers[0]:
        ax.hist(l1.flatten(), bins=bins, alpha=0.5, label=label, color=color)
    if layers[1]:
        ax.hist(l2.flatten(), bins=bins, alpha=0.5, label=label, color=color)
    if layers[2]:
        ax.hist(l3.flatten(), bins=bins, alpha=0.5, label=label, color=color)


def plot_weight_hist(weights, labels=None, suptitle=None, huevals=None, cmap=None, norm=None, colors=None):
    fig, ax = plt.subplots(3, 2, figsize=(15, 20))
    if labels is None:
        labels = [''] * len(weights)

    if colors is None:
        if huevals is None:
            colors = [None] * len(weights)
        else:
            colors = [cmap(norm(h)) for h in huevals]

    for weight, label, color in zip(weights, labels, colors):
        for row, rax in enumerate(ax):
            layers = (row == 0, row == 1, row == 2)
            for col, cax in enumerate(rax):
                w_idx = 0 if col == 0 else -1
                _plot_weight_hist(cax, weight[w_idx], layers=layers, label=label, color=color)
                cax.set(title=f"Layer {row} {'Initial' if w_idx == 0 else 'Final'}")
                cax.legend()
                cax.set(xlabel='Weight', ylabel='Count', xlim=(-1.5, 1.5))

    fig.suptitle(suptitle)
    plt.tight_layout()

    plt.show()

def animate_weight_hist(weights, labels=None, suptitle='', huevals=None, cmap=None, norm=None, colors=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 20), sharex=True)
    ax[0].set(ylabel='Count', xlim=(-1.5, 1.5), title='Layer 0')
    ax[1].set(ylabel='Count', xlim=(-1.5, 1.5), title='Layer 1')
    ax[2].set(xlabel='Weight', ylabel='Count', xlim=(-1.5, 1.5), title='Layer 2')
    fig.suptitle(f"{suptitle} - Epoch 0", fontsize=16, y=0.92)

    if labels is None:
        labels = [''] * len(weights)

    if colors is None:
        if huevals is None:
            colors = [None] * len(weights)
        else:
            colors = [cmap(norm(h)) for h in huevals]

    def get_frame(i):
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        ax[0].set(ylabel='Count', xlim=(-1.5, 1.5), title='Layer 0')
        ax[1].set(ylabel='Count', xlim=(-1.5, 1.5), title='Layer 1')
        ax[2].set(xlabel='Weight', ylabel='Count', xlim=(-1.5, 1.5), title='Layer 2')
        for weight, label, color in zip(weights, labels, colors):
            for row, rax in enumerate(ax):
                layers = (row == 0, row == 1, row == 2)
                _plot_weight_hist(rax, weight[i], layers=layers, label=label, color=color)
                rax.legend()
        fig.suptitle(f"{suptitle} - Epoch {i}", fontsize=16, y=0.92)

    anim = animation.FuncAnimation(fig, get_frame, frames=len(weights[0]), repeat=False)
    anim.save(f'{suptitle}.gif', writer='ffmpeg', fps=5)
    

if __name__ == '__main__':
    plt.style.use('ggplot')
    args = create_parser()
    args.func(args)