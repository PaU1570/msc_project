# courtesy of chatGPT

import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.style.use("ggplot")

# Function to read the file and plot data
def plot_data(filename, xaxis, yaxis):
    """
    Plot the data read from a text file outputted by Xyce.

    Args:
        filename: (str) The name of the file to read.
        xaxis: (str) The name of the column to be used as the x-axis. If None, 'TIME' will be used.
        yaxis: (list(str)) A list of strings with the names of the columns to plot against xaxis. If None, all other columns will be used.

    Returns:
        None
    """

    # Read the file using pandas. Skip the last line (contains ending message).
    data = pd.read_csv(filename, delim_whitespace=True, skipfooter=1, engine='python')
    # drop 'Index' column
    data = data.drop(['Index'], axis=1)
    print(data)

    # select the column to be used as the X-axis ('TIME' by default)
    xaxis = 'TIME' if xaxis is None else xaxis
    x = data[xaxis]

    # Plot columns in yaxis against time.
    for column in data.columns:
        if (column != xaxis) and ((yaxis is None) or (column in yaxis)):
            plt.plot(x, data[column], label=column)

    # Adding labels and title
    plt.xlabel(xaxis)
    plt.ylabel('Values')
    plt.title('Xyce Data Plot')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Main function to handle command line arguments
def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Plot data from a text file.')

    # Add argument for the filename
    parser.add_argument('filename', type=str, help='The path to the text file')
    parser.add_argument('-x', type=str, help='The name of the column to use as the X-axis')
    parser.add_argument('-y', type=str, help='The names of the columns to use as the Y-axis')

    # Parse the arguments
    args = parser.parse_args()
    # Call the plot function with the filename
    plot_data(args.filename, args.x, None if args.y is None else args.y.split(" "))

# Entry point of the script
if __name__ == "__main__":
    main()
