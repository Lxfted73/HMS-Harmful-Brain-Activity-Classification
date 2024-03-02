import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

df_test_eegs = pd.read_parquet("Data/test_eegs.parquet")

print("df_test_eegs.shape", df_test_eegs.shape)
print("df_test_eegs.head()\n", df_test_eegs.head)

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

# import numpy as np
# from scipy.signal import welch
# import matplotlib.pyplot as plt

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


# Define parameters
sampling_rate = 200  # Samples per second (SPS)
duration = 50  # Duration of EEG data in seconds
desired_frequency_resolution = 1  # Desired frequency resolution in Hz

# Calculate FFT length
fft_length = int(sampling_rate / desired_frequency_resolution)

# Choose window length (power of 2)
window_length = 2 ** int(np.ceil(np.log2(duration * sampling_rate)))

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

# Print the length of each dictionary
print("Length of psd_results:", len(psd_results))
print("Length of relative_power_results:", len(relative_power_results))
print("Length of peak_frequency_results:", len(peak_frequency_results))


# Iterate over dictionaries and print their lengths
for name, dictionary in [("psd_results", psd_results),
                          ("relative_power_results", relative_power_results),
                          ("peak_frequency_results", peak_frequency_results)]:
    print(f"Length of {name}: {len(dictionary)}")
#
# # Convert dictionaries to DataFrames
# psd_df = pd.DataFrame.from_dict(psd_results, orient='index', columns=['psd'])
relative_power_df = pd.DataFrame.from_dict(relative_power_results, orient='index',
                                           columns=['alpha', ' beta', 'delta', 'gamma', 'theta'])
# peak_frequency_df = pd.DataFrame.from_dict(peak_frequency_results, orient='index', columns=['peak_frequency'])
#
# # Concatenate DataFrames along the columns axis
df_test_eegs = pd.concat([df_test_eegs, relative_power_df], axis=1)
print("df_train_eegs.shape", df_test_eegs.shape)
print("df_train_eegs.head()\n", df_test_eegs.head)


# # Plot PSD for a specific electrode
# electrode_idx = 0  # Example electrode index
# plt.figure(figsize=(10, 6))
# plt.plot(f, psd_results[electrode_idx])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density (dB/Hz)')
# plt.title('Power Spectral Density for Electrode {}'.format(electrode_idx))
# plt.grid(True)
# plt.show()
#
# # Print relative power and peak frequency for the same electrode
# print("Relative Power:")
# print(relative_power_results[electrode_idx])
# print("\nPeak Frequency: {:.2f} Hz".format(peak_frequency_results[electrode_idx]))

