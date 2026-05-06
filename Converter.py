"""
Written by Paing Htet Kyaw - up2301555
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

def plot_cpu_usage(csv_file):
    """
    Plot CPU usage per core from a CSV log file.

    Args:
        csv_file (str): Path to the CPU usage CSV file.
    """
    try:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(csv_file)

        # Convert 'timestamp' column to datetime for readability
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Set 'timestamp' as the index
        df.set_index('timestamp', inplace=True)

        # Create a figure with specified size
        plt.figure(figsize=(10, 6))

        # Plot CPU usage for each core
        df.plot()

        # Add title and labels
        plt.title('CPU Usage Per Core Over Time')
        plt.ylabel('Usage (%)')
        plt.xlabel('Time')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add a legend to identify each core
        plt.legend(title='Cores')

        # Adjust layout to prevent label clipping
        plt.tight_layout()

        # Generate a unique output filename based on the input CSV
        output_file = csv_file.replace('.csv', '_plot.png')
        plt.savefig(output_file)
        print(f"Plot saved as {output_file}")

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
    except Exception as e:
        print(f"Error processing CSV file: {e}")

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Plot CPU usage from CSV log')
    parser.add_argument('csv_file', help='Path to the CPU usage CSV file')
    args = parser.parse_args()

    # Call the plotting function with the provided CSV file
    plot_cpu_usage(args.csv_file)
