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

Returns:
 - stft_data_results - A List containing the STFT transform for each of the Nodes
 
"""

class TimeFrequencyDomainFeatures:
    def __init__(self, data_file):
        self.df_eegs = pd.read_parquet(data_file)
        self.stft_data_results = None
        self.wavelet_coeffs_results = None

    def explain_stft_data_results(self):
        print("\nSTFT_data - List Length", len(self.stft_data_results))
        print("\nSTFT_data[0] - Length", len(self.stft_data_results[0]))

    def explain_wavelet_coeffs_results(self):
        print("\nWavelet_data - List Length", len(self.wavelet_coeffs_results))
        print("\nWavelet_data[0] - Length", len(self.wavelet_coeffs_results[0]))
    def check_missing_values(self):
        missing_values = self.df_eegs.isnull().sum()
        print("Missing Values:\n", missing_values)


    def print_data_info(self):
            print("df_eeegs.shape:", self.df_eegs.shape)
            print("df_eegs.head():\n", self.df_eegs.head())


    def compute_stft(self, print_graph_example=False):
        fs = 200
        self.stft_data_results = []
        for i in range(len(self.df_eegs.columns)-1):
            frequencies, time_segments, stft_data = stft(self.df_eegs.iloc[:, i], fs=fs, nperseg=fs * 2)
            self.stft_data_results.append(stft_data)
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
        return self.stft_data_results




    def compute_wavelet_decomposition(self, print_graph_example=False):
            wavelet_name = 'sym5'
            level = 5
            self.wavelet_coeffs_results = []
            for i in range(len(self.df_eegs.columns)-1):
                signal = self.df_eegs.iloc[:, i]
                coeffs = pywt.wavedec(signal, wavelet_name, level=level)
                self.wavelet_coeffs_results.append(coeffs)
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
            return self.wavelet_coeffs_results


eeg_analysis = TimeFrequencyDomainFeatures("/Users/thebowofapollo/PycharmProjects/HMS - Harmful Brain Activity Classification/train_eegs/568657.parquet")
eeg_analysis.check_missing_values()
eeg_analysis.print_data_info()
eeg_analysis.compute_stft(print_graph_example=True)
eeg_analysis.compute_wavelet_decomposition(print_graph_example=False)
eeg_analysis.explain_stft_data_results()
eeg_analysis.explain_wavelet_coeffs_results()
