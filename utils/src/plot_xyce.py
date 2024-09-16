# courtesy of chatGPT

import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.style.use("ggplot")

# Function to read the file and plot data
def plot_data(filename):
    # Read the file using pandas
    data = pd.read_csv(filename, delim_whitespace=True, skipfooter=1)  # Assuming space/tab delimited
    print(data)
    # Set the second column as TIME
    time = data.iloc[:, 1]

    # Plot all other columns against TIME
    for column in data.columns[2:]:
        plt.plot(time, data[column], label=column)

    # Adding labels and title
    plt.xlabel('TIME')
    plt.ylabel('Values')
    plt.title('Data Plot')
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

    # Parse the arguments
    args = parser.parse_args()

    # Call the plot function with the filename
    plot_data(args.filename)

# Entry point of the script
if __name__ == "__main__":
    main()
