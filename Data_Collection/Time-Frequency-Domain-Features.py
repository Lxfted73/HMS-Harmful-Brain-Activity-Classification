import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pywt


"""
Fix Wavelets 
"""
print_graph_stft = False
print_graph_wavelet = True

# Read Parquet file into a DataFrame
df_test_eegs = pd.read_parquet("Data/test_eegs.parquet")

#Check for Missing Values
missing_values = df_test_eegs.isnull().sum()
print("Missing Values:\n", missing_values)

#no missing values

#Print Data Shape
print("df_test_eegs.shape:", df_test_eegs.shape)
print("df_test_eegs.head():\n", df_test_eegs.head())

# Sampling Frequency
fs = 200
signal = df_test_eegs.iloc[:, 0]
stft_data_results = []
# Compute STFT
for i in range(len(df_test_eegs.columns)-1):
    frequencies, time_segments, stft_data = stft(df_test_eegs.iloc[:, i], fs=fs, nperseg=fs * 2)
    stft_data_results.append(stft_data)

    #Return stdt_data_results

#  Plot STFT
if print_graph_stft:
    # Sampling Frequency
    fs = 200
    signal = df_test_eegs.iloc[:, 0]
    # Compute STFT
    frequencies, time_segments, stft_data = stft(df_test_eegs.iloc[:, 0], fs=fs, nperseg=fs*2)
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


# Compute wavelets calculation
# Define wavelet parameters
wavelet_name = 'sym5'  # Symlet 5 wavelet
level = 5  # Decoposition level
wavelet_coeffs_results = []

for i in range(len(df_test_eegs.columns)-1)
    signal = df_test_eegs.iloc[:, 1]
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    wavelet_coeffs_results.append(coeffs)

# Return wavelet_coeefs_results

if print_graph_wavelet:
    # Define wavelet parameters
    wavelet_name = 'sym5'  # Symlet 5 wavelet
    level = 5  # Decoposition level

    signal = df_test_eegs.iloc[:, 1]
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)

    #Return Coeffs


    # Plot wavelet coefficients
    print_graph_wavelet = True  # Set to True if you want to plot
    if print_graph_wavelet:
        plt.figure(figsize=(10, 10))
        for i in range(len(coeffs)):
            plt.subplot(level+1, 1, i+1)
            plt.plot(coeffs[i])
            plt.title(f'Detail {i+1}' if i > 0 else 'Approximation')
        plt.tight_layout()
        plt.show()


