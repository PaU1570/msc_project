import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
from matplotlib import colormaps
import seaborn as sns

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


def plot_summary(args):

    if not os.path.isfile(args.input):
        print(f"Error: The file {args.input} does not exist.")
        exit(1)

    xscale = 'linear' if args.scale in ['linear', 'log-lin'] else 'log'
    yscale = 'linear' if args.scale in ['linear', 'lin-log'] else 'log'

    title = args.title if args.title is not None else f'{args.input}'

    data = pd.read_csv(args.input)
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
        data = data.reset_index(drop=True)

    if not args.all:
        if args.x not in data.columns :
            print(f"Error: {args.x} is not a column in the data.")
            exit(1)
        if args.y not in data.columns :
            print(f"Error: {args.y} is not a column in the data.")
            exit(1)

        fig, ax = plt.subplots(figsize=(8, 6))

        unique_types = data[args.hue].unique()
        
        if data[args.hue].dtype == 'float64':
            norm = clr.LogNorm() if args.huescale == 'log' else clr.Normalize()       
            scatter = ax.scatter(data[args.x], data[args.y], c=data[args.hue], cmap='plasma', norm=norm)
            cbar = plt.colorbar(scatter)
            cbar.set_label(args.hue)
        else:
            #colors = sns.color_palette('husl', len(unique_types))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_types)))
            for unique_type, color in zip(unique_types, colors):
                type_data = data[data[args.hue] == unique_type]
                ax.scatter(type_data[args.x], type_data[args.y], label=unique_type, color=color)
            ax.legend(title=args.hue)

        ax.set(xlabel=args.x, ylabel=args.y, xscale=xscale, yscale=yscale, title=title, aspect=args.aspect, xlim=args.xlim, ylim=args.ylim)
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig, facecolor=fig.get_facecolor())
        plt.show()

    if args.all:
        cols = ['stepSize','pulseWidth','onOffRatio','accuracy', 'A_LTP', 'A_LTD']
        g = sns.pairplot(data, hue=args.hue, vars=cols)
        g.fig.suptitle(title)
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig)
        plt.show()

def plot_epochs(args):
    pass

if __name__ == '__main__':
    plt.style.use('ggplot')
    args = create_parser()
    args.func(args)