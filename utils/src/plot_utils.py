import argparse

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

    parser_epochs = subparsers.add_parser('epochs', help='Plot training evolution over epochs', parents=[common_parser])
    parser_epochs.add_argument('-y', type=str, help='Y axis value', choices=['accuracy', 'read_latency', 'write_latency', 'read_energy', 'write_energy'], default='accuracy')

    args = parser.parse_args()

    # check summary args
    if args.subcommand == 'summary':
        if (args.x or args.y) and args.all:
            parser.error("Cannot specify -x and -y when using --all")
        if (args.x and not args.y) or (args.y and not args.x):
            parser.error("Must specify both -x and -y")
        if not (args.x or args.y) and not args.all:
            parser.error("Must specify either -x and -y or --all")

    return args

