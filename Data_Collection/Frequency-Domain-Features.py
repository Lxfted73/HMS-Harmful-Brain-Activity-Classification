import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

"""
To-do: Plot one type of case in EEG Data

"""


# Read Parquet file into a DataFrame
df_test_eegs = pd.read_parquet("Data/test_eegs.parquet")


# Calculate the Power Spectral Density (PSD)
num_channels = df_test_eegs.shape[1]  # Number of EEG channels

# Calculate PSD for each electrode node
psd_results = []
fs = 200 # See HMS Data Page under train_eegs/
for i in range(num_channels-1):
    # Extract EEG signal data from the ith electrode node
    eeg_signal = df_test_eegs.iloc[:, i].values

    # Calculate PSD using Welch's Method
    frequencies, psd = welch(eeg_signal, fs=fs, nperseg=fs*2)

    # Store Results
    psd_results.append((frequencies, psd))

#Visualize PSD for the first electrode
# plt.figure()
# plt.specgram(psd_results, Fs=400, NFFT=400, noverlap=200)
# plt.title(f"PSD for Electrode Node 1")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power/Frequency (db/Hz)")
# plt.grid(True)
# plt.show()

#
plt.figure()
for i in range(num_channels-1):
    x_axis = psd_results[i][0][:]
    y_axis = psd_results[i][1][:]
    plt.semilogy(x_axis, y_axis, label=f"Channel {i+1}")
plt.title(f"PSD for Electrode Node 1")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (db/Hz)")
plt.legend()
plt.grid(True)


plt.show()




