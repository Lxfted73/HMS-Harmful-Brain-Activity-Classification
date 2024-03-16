import np
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

Returns: 
1. df.eeg_results contains a dictionary with mean, min, max, std, var, rms values for each electrode including ECG
2. df.spectrogram_results contains a dictionary with mean, min, max, std, var, rms values for each electrode including spectrogram
"""

class Time_Domain_Features:
    def __init__(self, eegs_data_file, spectrogram_data_file):
        self.df_eegs = pd.read_parquet(eegs_data_file)
        self.df_eegs_results = {}
        self.df_eegs_mean_list = []
        self.df_eegs_min_list = []
        self.df_eegs_max_list = []
        self.df_eegs_std_list = []
        self.df_eegs_var_list = []
        self.df_eegs_rms_list = []
        self.df_spectrogram = pd.read_parquet(spectrogram_data_file)
        self.df_spectrogram_results = {}
        self.df_spectrogram_mean_list = []
        self.df_spectrogram_min_list = []
        self.df_spectrogram_max_list = []
        self.df_spectrogram_std_list = []
        self.df_spectrogram_var_list = []
        self.df_spectrogram_rms_list = []

    def analyze_eeg_data(self,print_console = False, print_graph=False):

            for i in self.df_eegs.columns:
            # Check for Missing Values


                # Plot one electrode of data:
                sampling_frequency = 200  # samples per second
                num_samples = 10000  # for example, plot 10000 samples
                time_per_sample = 1 / sampling_frequency  # seconds per sample
                sample_indices = np.arange(num_samples)
                time_seconds = sample_indices * time_per_sample


                mean_values = self.df_eegs[:][i].mean()
                self.df_eegs_mean_list.append(mean_values)
                self.df_eegs_results.update({'mean': self.df_eegs_mean_list})
                min_values = self.df_eegs[:][i].min()
                self.df_eegs_min_list.append(min_values)
                self.df_eegs_results.update({'min': self.df_eegs_min_list})
                max_values = self.df_eegs[:][i].max()
                self.df_eegs_max_list.append(max_values)
                self.df_eegs_results.update({'max': self.df_eegs_max_list})
                std_values = self.df_eegs[:][i].std()
                self.df_eegs_std_list.append(std_values)
                self.df_eegs_results.update({'std': self.df_eegs_std_list})
                variance_values = self.df_eegs[:][i].var()
                self.df_eegs_var_list.append(variance_values)
                self.df_eegs_results.update({'var': self.df_eegs_var_list})
                rms_values = np.sqrt(np.mean(self.df_eegs[:][i] ** 2))
                self.df_eegs_rms_list.append(rms_values)
                self.df_eegs_results.update({'rms': self.df_eegs_rms_list})

            if print_console:
                missing_values = self.df_eegs.isnull().sum()
                print("Missing Values:\n", missing_values)
                print("self.df_eegs.shape:", self.df_eegs.shape)
                print("self.df_eegs.head():\n", self.df_eegs.head())
                print("Mean Values:\n", self.df_eegs_mean_list)
                print("Min Values:\n", self.df_eegs_min_list)
                print("Max Values:\n", self.df_eegs_max_list)
                print("Standard Deviation:\n", self.df_eegs_std_list)
                print("Variance:\n", self.df_eegs_var_list)
                print("RMS:\n", self.df_eegs_rms_list)

            if print_graph:
                plt.figure(figsize=(10, 6))
                plt.plot(time_seconds, self.df_eegs['Fp1'])
                plt.title("Time Domain Values of Fp1")
                plt.xlabel('Seconds')
                plt.ylabel('Voltage (units)')
                plt.show()



    def analyze_spectrogram_data(self, print_console=False, print_graph=False):
        # Plot one electrode of data:
        sampling_frequency = 200  # samples per second
        num_samples = 10000  # for example, plot 10000 samples
        time_per_sample = 1 / sampling_frequency  # seconds per sample
        sample_indices = np.arange(num_samples)
        time_seconds = sample_indices * time_per_sample

        print (self.df_spectrogram.head())
        for i in self.df_spectrogram.columns:
            mean_values = self.df_spectrogram[:][i].mean()
            self.df_spectrogram_mean_list.append(mean_values)
            self.df_spectrogram_results.update({'mean': self.df_spectrogram_mean_list})
            min_values = self.df_spectrogram[:][i].min()
            self.df_spectrogram_min_list.append(min_values)
            self.df_spectrogram_results.update({'min': self.df_spectrogram_min_list})
            max_values = self.df_spectrogram[:][i].max()
            self.df_spectrogram_max_list.append(max_values)
            self.df_spectrogram_results.update({'max': self.df_spectrogram_max_list})
            std_values = self.df_spectrogram[:][i].std()
            self.df_spectrogram_std_list.append(std_values)
            self.df_spectrogram_results.update({'std': self.df_spectrogram_std_list})
            variance_values = self.df_spectrogram[:][i].var()
            self.df_spectrogram_var_list.append(variance_values)
            self.df_spectrogram_results.update({'var': self.df_spectrogram_var_list})
            rms_values = np.sqrt(np.mean(self.df_spectrogram[:][i] ** 2))
            self.df_spectrogram_rms_list.append(rms_values)
            self.df_spectrogram_results.update({'rms': self.df_spectrogram_rms_list})

        if print_console:
            print("Mean Values:\n", self.df_spectrogram_mean_list)
            print("Min Values:\n", self.df_spectrogram_min_list)
            print("Max Values:\n", self.df_spectrogram_max_list)
            print("Standard Deviation:\n", self.df_spectrogram_std_list)
            print("Variance:\n", self.df_spectrogram_var_list)
            print("RMS:\n", self.df_spectrogram_rms_list)

        if print_graph:
            # Create a figure and subplots
            fig, axs = plt.subplots(6, figsize=(8, 12), sharex=True)

            # Plot mean values
            axs[0].plot(self.df_spectrogram_mean_list, marker='o', linestyle='-', color='blue')
            axs[0].set_title('Mean Values')
            axs[0].set_ylabel('Mean')

            # Plot min values
            axs[1].plot(self.df_spectrogram_min_list, marker='o', linestyle='-', color='orange')
            axs[1].set_title('Min Values')
            axs[1].set_ylabel('Min')

            # Plot max values
            axs[2].plot(self.df_spectrogram_max_list, marker='o', linestyle='-', color='green')
            axs[2].set_title('Max Values')
            axs[2].set_ylabel('Max')

            # Plot standard deviation values
            axs[3].plot(self.df_spectrogram_std_list, marker='o', linestyle='-', color='red')
            axs[3].set_title('Standard Deviation Values')
            axs[3].set_ylabel('Std')

            # Plot variance values
            axs[4].plot(self.df_spectrogram_var_list, marker='o', linestyle='-', color='purple')
            axs[4].set_title('Variance Values')
            axs[4].set_ylabel('Var')

            # Plot RMS values
            axs[5].plot(self.df_spectrogram_rms_list, marker='o', linestyle='-', color='brown')
            axs[5].set_title('RMS Values')
            axs[5].set_ylabel('RMS')

            # Set common x-axis label
            axs[-1].set_xlabel('Index')

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the plot
            plt.show()


tdf = Time_Domain_Features("train_eegs/568657.parquet", "train_spectrograms/924234.parquet")
tdf.analyze_eeg_data(print_graph=False, print_console=True)
tdf.analyze_spectrogram_data(print_graph=False, print_console=True)