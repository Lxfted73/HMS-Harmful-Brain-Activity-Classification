import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

"""
Script for analyzing EEG and spectrogram data.

This script reads EEG and spectrogram data from Parquet files, calculates various statistics such as mean, min, max, standard deviation, variance, and RMS values, and plots EEG data for visualization.

The script is divided into the following sections:
1. Importing necessary libraries.
2. Reading EEG and spectrogram data from Parquet files.
3. Analyzing EEG data:
   - Plotting EEG data.
   - Calculating mean, min, max, standard deviation, variance, and RMS values.
4. Analyzing spectrogram data:
   - Calculating mean, min, max, standard deviation, variance, and RMS values.
5. Displaying analysis results.
"""

class Time_Domain_Features:
    def __init__(self, eegs_data_file, spectrogram_data_file):
        self.df_eegs = pd.read_parquet(eegs_data_file)
        self.df_spectrogram = pd.read_parquet(spectrogram_data_file)

    def analyze_eeg_data(self,print_console = False, print_graph=False):

        # Check for Missing Values
        missing_values = self.df_eegs.isnull().sum()
        if print_console:
            print("Missing Values:\n", missing_values)

            # Print Data Shape
            print("self.df_eegs.shape:", self.df_eegs.shape)
            print("self.df_eegs.head():\n", self.df_eegs.head())

        # Plot one electrode of data:
        sampling_frequency = 200  # samples per second
        num_samples = 10000  # for example, plot 10000 samples
        time_per_sample = 1 / sampling_frequency  # seconds per sample
        sample_indices = np.arange(num_samples)
        time_seconds = sample_indices * time_per_sample
        if print_graph:
            plt.figure(figsize=(10, 6))
            plt.plot(time_seconds, self.df_eegs['Fp1'])
            plt.title("Frequency of Fp1")
            plt.xlabel('Seconds')
            plt.ylabel('Voltage (units)')
            plt.show()

            # Calculate Mean Value
            mean_values = self.df_eegs.mean()
            print("Mean Values:\n", mean_values)

            # Calculate Min and Max Values
            min_values = self.df_eegs.min()
            max_values = self.df_eegs.max()
            print("Min Values:\n", min_values)
            print("Max Values:\n", max_values)

            # Calculate Standard Deviation, Variance, and RMS Values
            std_values = self.df_eegs.std()
            variance_values = self.df_eegs.var()
            rms_values = np.sqrt(np.mean(self.df_eegs ** 2))

            print("Standard Deviation:\n", std_values)
            print("Variance:\n", variance_values)
            print("RMS:\n", rms_values)


    def analyze_spectrogram_data(self, print_console=False, print_graph=False):
        mean_values = self.df_spectrogram.mean()
        if print_console:
            print("Mean Values:\n", mean_values)

        min_values = self.df_spectrogram.min()
        max_values = self.df_spectrogram.max()
        if print_console:
            print("Min Values:\n", min_values)
            print("\nMax Values:\n", max_values)

        std_values = self.df_spectrogram.std()
        variance_values = self.df_spectrogram.var()
        rms_values = np.sqrt(np.mean(self.df_spectrogram**2))
        if print_console:
            print("Standard Deviation:\n", std_values)
            print("\n\nVariance:\n", variance_values)
            print("\n\nRMS:\n", rms_values)
        return

tdf = Time_Domain_Features("Data/test_eegs.parquet", "Data/test_spectogram.parquet")
tdf.analyze_eeg_data(print_graph=True)
tdf.analyze_spectrogram_data(print_graph=True
                             )