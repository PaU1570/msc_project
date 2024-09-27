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

    data = pd.read_csv(args.input)

    if not args.all:
        if args.x not in data.columns :
            print(f"Error: {args.x} is not a column in the data.")
            exit(1)
        if args.y not in data.columns :
            print(f"Error: {args.y} is not a column in the data.")
            exit(1)

        fig, ax = plt.subplots(figsize=(8, 6))

        unique_devices = data['device_id'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_devices)))

        for device, color in zip(unique_devices, colors):
            device_data = data[data['device_id'] == device]
            ax.scatter(device_data[args.x], device_data[args.y], label=device, color=color)

        ax.legend(title='Device ID')
        ax.set(xlabel=args.x, ylabel=args.y, xscale=xscale, yscale=yscale)
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig, facecolor=fig.get_facecolor())
        plt.show()

    if args.all:
        cols = ['stepSize','pulseWidth','onOffRatio','accuracy', 'A_LTP', 'A_LTD']
        sns.pairplot(data, hue='device_id', vars=cols)
        if args.savefig is not None:
            plt.tight_layout()
            plt.savefig(args.savefig)
        plt.show()