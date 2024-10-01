import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

plt.style.use('ggplot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot summary')
    parser.add_argument('input', type=str, help='Summary file output by run_neurosim_to_csv.py')
    parser.add_argument('-x', type=str, help='X axis value')
    parser.add_argument('-y', type=str, help='Y axis value')
    parser.add_argument('--all', action='store_true', help='Plot all pairs of data points')
    parser.add_argument('--savefig', type=str, help='Save the figure to a file', default=None)
    parser.add_argument('--scale', type=str, choices=['linear', 'log-log', 'log-lin', 'lin-log'], default='linear', help='Scale of the axes')
    parser.add_argument('--title', type=str, help='Title of the plot', default=None)
    parser.add_argument('--hue', type=str, help='Which attribute to use for coloring the data points', default='device_id')
    parser.add_argument('--aspect', type=str, help='Aspect ratio of the plot', default='auto')
    parser.add_argument('--xlim', type=float, nargs=2, help='X axis limits', default=None)
    parser.add_argument('--ylim', type=float, nargs=2, help='Y axis limits', default=None)
    parser.add_argument('--filter', type=str, help='Filter the data based on a column', default=None, nargs=2)
    args = parser.parse_args()

    if (args.x or args.y) and args.all:
        parser.error("Cannot specify -x and -y when using --all")
    if (args.x and not args.y) or (args.y and not args.x):
        parser.error("Must specify both -x and -y")
    if not (args.x or args.y) and not args.all:
        parser.error("Must specify either -x and -y or --all")

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
            scatter = ax.scatter(data[args.x], data[args.y], c=data[args.hue], cmap='plasma')
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