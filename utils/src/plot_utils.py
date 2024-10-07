import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
from matplotlib import colormaps
import seaborn as sns

from run_neurosim_to_csv import get_data_from_file

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
    common_parser.add_argument('--xlim', type=float, nargs=2, help='X axis limits', default=None)
    common_parser.add_argument('--ylim', type=float, nargs=2, help='Y axis limits', default=None)
    common_parser.add_argument('--scale', type=str, choices=['linear', 'log-log', 'log-lin', 'lin-log'], default='linear', help='Scale of the axes')
    common_parser.add_argument('--aspect', type=str, help='Aspect ratio of the plot', default='auto')
    # data selection arguments
    common_parser.add_argument('--filter', type=str, help='Filter the data based on a column', default=None, nargs=2)
    common_parser.add_argument('--hue', type=str, help='Which attribute to use for coloring the data points', default='device_id')
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
    parser_epochs.add_argument('-y', type=str, help='Y axis value', choices=['accuracy', 'read_latency', 'write_latency', 'read_energy', 'write_energy'], default='accuracy')
    parser_epochs.set_defaults(func=plot_epochs)

    args = parser.parse_args()

    # check summary args
    if args.subcommand == 'summary':
        if (args.x or args.y) and args.all:
            parser.error("Cannot specify -x and -y when using --all")
        if (args.x and not args.y) or (args.y and not args.x):
            parser.error("Must specify both -x and -y")
        if not (args.x or args.y) and not args.all:
            parser.error("Must specify either -x and -y or --all")
    elif args.subcommand == 'epochs':
        pass # no checks yet
    else:
        parser.error("Please specify a subcommand")

    return args

def filter_data(data, filter_column, filter_value, reset_index=True):
    """
    Filter the data based on a column and a value.

    Args:
        data (pd.DataFrame): The data to filter
        filter_column (str): The column to filter on
        filter_value (int, float, str): The value to filter on

    Returns:
        pd.DataFrame: The filtered data
    """
    #TODO: add support for multiple filters
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

    data['avgAccuracy'] = data.groupby(['device_id', 'test_date', 'test_time'])['accuracy'].transform('mean')
    data['stdAccuracy'] = data.groupby(['device_id', 'test_date', 'test_time'])['accuracy'].transform('std')
    data.drop_duplicates(subset=['device_id', 'test_date', 'test_time'], inplace=True)
    data.drop(columns=['accuracy'], inplace=True)
    data.rename(columns={'avgAccuracy': 'accuracy'}, inplace=True)
    return data

def plot_summary(args):

    isFile = os.path.isfile(args.input)
    isDir = os.path.isdir(args.input)
    if not isFile and not isDir:
        print(f"Error: The file or directory {args.input} does not exist.")
        exit(1)

    xscale = 'linear' if args.scale in ['linear', 'log-lin'] else 'log'
    yscale = 'linear' if args.scale in ['linear', 'lin-log'] else 'log'

    title = args.title if args.title is not None else f'{args.input}'

    if isFile:
        data = pd.read_csv(args.input)
    elif isDir:
        data = read_average_data(args.input)

    if args.filter is not None:
        data = filter_data(data, args.filter[0], args.filter[1])

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

        unique_types = data[args.hue].unique()
        
        if data[args.hue].dtype == 'float64':
            norm = clr.LogNorm() if args.huescale == 'log' else clr.Normalize()       
            cmap = plt.cm.plasma
            colors = cmap(norm(data[args.hue].values))

            cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
            cbar.set_label(args.hue)

            if xerr is None:
                xerr = np.zeros(len(data[args.x]))
            if yerr is None:
                yerr = np.zeros(len(data[args.y]))

            for x, y, ex, ey, color in zip(data[args.x], data[args.y], xerr, yerr, colors):
                ax.errorbar(x, y, xerr=ex, yerr=ey, fmt='o', capsize=2, color=color, ecolor=color)

        else:
            #colors = sns.color_palette('husl', len(unique_types))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
            for unique_type, color in zip(unique_types, colors):
                type_data = data[data[args.hue] == unique_type]
                ax.errorbar(type_data[args.x], type_data[args.y], xerr=xerr, yerr=yerr, fmt='o', capsize=2, color=color, ecolor=color, label=unique_type)
            ax.legend(title=args.hue)

        ax.set(xlabel=args.x, ylabel=args.y, xscale=xscale, yscale=yscale, title=title, aspect=args.aspect, xlim=args.xlim, ylim=args.ylim)
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig, facecolor=fig.get_facecolor())
        plt.show()

    if args.all:
        cols = ['stepSize','pulseWidth','onOffRatio','accuracy', 'A_LTP', 'A_LTD']
        g = sns.pairplot(data, hue=args.hue, vars=cols)
        g.figure.suptitle(title)
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig)
        plt.show()

def plot_epochs(args):
    files = os.listdir(args.input)
    files = [f for f in files if f.endswith('.dat')]

    xscale = 'linear' if args.scale in ['linear', 'log-lin'] else 'log'
    yscale = 'linear' if args.scale in ['linear', 'lin-log'] else 'log'
    
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
        data = filter_data(data, args.filter[0], args.filter[1], reset_index=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    if data[args.hue].dtype == 'float64':
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

    ax.set(xlabel='Epoch', ylabel=args.y, xscale=xscale, yscale=yscale, title=title, aspect=args.aspect, xlim=args.xlim, ylim=args.ylim)

    if args.savefig is not None:
        plt.tight_layout()
        plt.savefig(args.savefig)
    plt.show()

if __name__ == '__main__':
    plt.style.use('ggplot')
    args = create_parser()
    args.func(args)