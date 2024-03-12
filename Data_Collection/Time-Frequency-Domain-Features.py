import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pywt

"""
This Python script analyzes EEG data by computing Short-Time Fourier Transform (STFT) and performing wavelet decomposition using the PyWavelets library.

Parameters Configuration:
- print_graph_stft: Set to True if you want to print STFT graphs.
- print_graph_wavelet: Set to True if you want to print wavelet decomposition graphs.

Reading Data:
- EEG data is read from a Parquet file into a DataFrame (df_eegs).

Data Analysis:
- Checks for missing values in the DataFrame.
- Prints the shape and head of the DataFrame.
- Computes STFT for each EEG signal using scipy.signal.stft.
- Computes wavelet decomposition for each EEG signal using PyWavelets' wavedec.

Plotting:
- If print_graph_stft is enabled, it plots the STFT magnitude.
- If print_graph_wavelet is enabled, it plots the wavelet coefficients.
"""

class tfdf:
    def __init__(self, data_file):
        self.df_eegs = pd.read_parquet(data_file)
        self.stft_data_results = None

    def check_missing_values(self):
        missing_values = self.df_eegs.isnull().sum()
        print("Missing Values:\n", missing_values)


    def print_data_info(self):
            print("df_eegs.shape:", self.df_eegs.shape)
            print("df_eegs.head():\n", self.df_eegs.head())


    def compute_stft(self, print_graph=False):
        fs = 200
        stft_data_results = []
        for i in range(len(self.df_eegs.columns)-1):
            frequencies, time_segments, stft_data = stft(self.df_eegs.iloc[:, i], fs=fs, nperseg=fs * 2)
            stft_data_results.append(stft_data)
        if print_graph:
            # Sampling Frequency
            fs = 200
            signal = self.df_eegs.iloc[:, 0]
            # Compute STFT
            frequencies, time_segments, stft_data = stft(self.df_eegs.iloc[:, 0], fs=fs, nperseg=fs * 2)
            #  Plot STFT
            plt.pcolormesh(time_segments, frequencies, np.abs(stft_data), shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.xlim(0, 50)
            plt.yscale('log')
            plt.ylim(0.1, 2)
            plt.colorbar(label='Magnitude')
            plt.show()
        return stft_data_results




    def compute_wavelet_decomposition(self, print_graph=False):
            wavelet_name = 'sym5'
            level = 5
            wavelet_coeffs_results = []
            for i in range(len(self.df_eegs.columns)-1):
                signal = self.df_eegs.iloc[:, 1]
                coeffs = pywt.wavedec(signal, wavelet_name, level=level)
                wavelet_coeffs_results.append(coeffs)
            if print_graph:
                # Define wavelet parameters
                wavelet_name = 'sym5'  # Symlet 5 wavelet
                level = 5  # Decoposition level

                signal = self.df_eegs.iloc[:, 1]
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(signal, wavelet_name, level=level)

                # Return Coeffs

                # Plot wavelet coefficients
                plt.figure(figsize=(10, 10))
                for i in range(len(coeffs)):
                    plt.subplot(level + 1, 1, i + 1)
                    plt.plot(coeffs[i])
                    plt.title(f'Detail {i + 1}' if i > 0 else 'Approximation')
                plt.tight_layout()
                plt.show()

eeg_analysis = tfdf("Data/eegs.parquet")
eeg_analysis.check_missing_values()
eeg_analysis.print_data_info()
eeg_analysis.compute_stft(print_graph=True)
eeg_analysis.compute_wavelet_decomposition(print_graph=False)

