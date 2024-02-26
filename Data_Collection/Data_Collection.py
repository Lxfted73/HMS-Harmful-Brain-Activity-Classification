import pandas as pd

# Read Parquet file into a DataFrame
df_test_eegs = pd.read_parquet("Data/test_eegs.parquet")
# df_test_spectograms = pd.read_parquet("/Data/test_spectogram.parquet")
# df_train_eegs = pd.read_parquet("Data/train_eegs.parquet")
# df_train_spectograms = pd.read_parquet("Data/train_spectogram.parquet")

pd.set_option('display.max_columns', None)

print("df_train_eegs.shape", df_test_eegs.shape)
print("df_train_eegs.head()\n", df_test_eegs.head)

electrodes = df_test_eegs.columns[:-1]  # Exclude the last column (EKG)
electrode_locations = df_test_eegs[electrodes]

# Extract signal amplitude features
signal_amplitudes_mean = df_test_eegs[electrodes].mean(axis=0)
signal_amplitudes_max = df_test_eegs[electrodes].max(axis=0)
signal_amplitudes_min = df_test_eegs[electrodes].min(axis=0)
signal_amplitudes_variance = df_test_eegs[electrodes].var(axis=0)

# # Print the calculated features
# print("Mean Signal Amplitudes:")
# print(signal_amplitudes_mean)
# print("\nMaximum Signal Amplitudes:")
# print(signal_amplitudes_max)
# print("\nMinimum Signal Amplitudes:")
# print(signal_amplitudes_min)
# print("\nSignal Amplitudes Variance:")
# print(signal_amplitudes_variance)

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Assume df_test_eegs contains preprocessed EEG data with shape (n_samples, n_channels)

# Define frequency bands
freq_bands = {'delta': (0.5, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'gamma': (30, 100)}

# Define sampling frequency and window length for spectral analysis
fs = 200  # Sample rate (Hz)
window_length = 4 * fs  # 4-second window length

# Initialize dictionaries to store results
psd_results = {}
relative_power_results = {}
peak_frequency_results = {}

# Perform spectral analysis for each electrode
for channel in range(df_test_eegs.shape[1]):
    # Compute PSD using Welch's method
    f, psd = welch(df_test_eegs.iloc[:, channel], fs=fs, nperseg=window_length)

    # Calculate relative power in each frequency band
    total_power = np.sum(psd)
    relative_power = {band: np.sum(psd[(f >= f_band[0]) & (f < f_band[1])]) / total_power
                      for band, f_band in freq_bands.items()}

    # Extract peak frequency
    peak_frequency = f[np.argmax(psd)]

    # Store results
    psd_results[channel] = psd
    relative_power_results[channel] = relative_power
    peak_frequency_results[channel] = peak_frequency


# Convert dictionaries to DataFrames
psd_df = pd.DataFrame.from_dict(psd_results, orient='index', columns=['psd'])
relative_power_df = pd.DataFrame.from_dict(relative_power_results, orient='index', columns=['relative_power'])
peak_frequency_df = pd.DataFrame.from_dict(peak_frequency_results, orient='index', columns=['peak_frequency'])

# Concatenate DataFrames along the columns axis
result_df = pd.concat([psd_df, relative_power_df, peak_frequency_df], axis=1)


# Plot PSD for a specific electrode
electrode_idx = 0  # Example electrode index
plt.figure(figsize=(10, 6))
plt.plot(f, psd_results[electrode_idx])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('Power Spectral Density for Electrode {}'.format(electrode_idx))
plt.grid(True)
plt.show()

# Print relative power and peak frequency for the same electrode
print("Relative Power:")
print(relative_power_results[electrode_idx])
print("\nPeak Frequency: {:.2f} Hz".format(peak_frequency_results[electrode_idx]))

