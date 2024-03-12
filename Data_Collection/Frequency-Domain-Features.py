import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

"""
This script reads EEG data from a Parquet file, calculates the Power Spectral Density (PSD) using Welch's method,
and analyzes the PSD to extract power values within specific frequency bands associated with brainwaves (delta, theta, alpha, beta, gamma).
It then plots the PSD and the power values for each frequency band.

Summary of Steps:
1. Import necessary libraries: pandas, numpy, scipy.signal, and matplotlib.pyplot.
2. Set a flag to control whether to plot graphs or not.
3. Read EEG data from a Parquet file into a DataFrame.
4. Calculate PSD for each electrode using Welch's method.
5. If plotting is enabled, visualize the PSD for each electrode.
6. Define frequency ranges for different brainwave bands.
7. Extract frequencies and corresponding power values within each frequency band.
8. Concatenate frequency and power values for each frequency band.
9. If plotting is enabled, plot power values for each frequency band.

Note: Inline comments are provided throughout the code to explain each step and provide context.
"""

plot_graphs = True


# Read Parquet file into a DataFrame
df_eegs = pd.read_parquet("Data/eegs.parquet")


# Calculate the Power Spectral Density (PSD)
num_channels = df_eegs.shape[1]  # Number of EEG channels

# Calculate PSD for each electrode node
psd_results = []
fs = 200 # See HMS Data Page under train_eegs/
# -1 to exclude EKGs
for i in range(num_channels-1):
    # Extract EEG signal data from the ith electrode node
    eeg_signal = df_eegs.iloc[:, i].values

    # Calculate PSD using Welch's Method
    frequencies, psd = welch(eeg_signal, fs=fs, nperseg=fs*2)

    # Store Results
    psd_results.append((frequencies, psd))


if plot_graphs:
    # Visualize PSD for the first electrode
    plt.figure()
    x_axis = psd_results[0][0]
    y_axis = psd_results[0][1]
    plt.bar(x_axis, y_axis)
    plt.title(f"PSD for Electrode Node 1")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (db/Hz)")
    plt.grid(True)
    plt.show()

    #
    plt.figure()
    # -1 to exclude EKG
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


# print("psd_results\n", psd_results)
print("len(psd_results)\n:", len(psd_results))
print("len(psd_results[0])\n", len(psd_results[0]))
print("len(psd_results[0][0])\n", len(psd_results[0][0]))
print("len(psd_results[0][1])\n", len(psd_results[0][1]))
print("psd_results[0][0]\n", psd_results[0][0])
print("psd_results[0][1]\n", psd_results[0][1])


# Duplicated for sake of simplicity in reading below
frequencies = psd_results[0][0]
power_values = psd_results[0][1]
delta_range = (0.5, 4)
theta_range = (4, 8)
alpha_range = (8, 13)
beta_range = (13, 30)
gamma_range = (30, np.max(frequencies))

delta_frequencies_results = []
theta_frequencies_results = []
alpha_frequencies_results = []
beta_frequencies_results = []
gamma_frequencies_results = []

delta_power_results = []
theta_power_results = []
alpha_power_results = []
beta_power_results = []
gamma_power_results = []

for i in range(num_channels-1):
    frequencies = psd_results[i][0]
    power_values = psd_results[i][1]

    delta_frequencies_results.append(frequencies[(frequencies >= delta_range[0]) & (frequencies < delta_range[1])])
    theta_frequencies_results.append(frequencies[(frequencies >= theta_range[0]) & (frequencies < theta_range[1])])
    alpha_frequencies_results.append(frequencies[(frequencies >= alpha_range[0]) & (frequencies < alpha_range[1])])
    beta_frequencies_results.append(frequencies[(frequencies >= beta_range[0]) & (frequencies < beta_range[1])])
    gamma_frequencies_results.append(frequencies[(frequencies >= gamma_range[0]) & (frequencies <= gamma_range[1])])

    delta_power_results.append(power_values[(frequencies >= delta_range[0]) & (frequencies < delta_range[1])])
    theta_power_results.append(power_values[(frequencies >= theta_range[0]) & (frequencies < theta_range[1])])
    alpha_power_results.append(power_values[(frequencies >= alpha_range[0]) & (frequencies < alpha_range[1])])
    beta_power_results.append(power_values[(frequencies >= beta_range[0]) & (frequencies < beta_range[1])])
    gamma_power_results.append(power_values[(frequencies >= gamma_range[0]) & (frequencies <= gamma_range[1])])

print("len(delta_frequency_results)", len(delta_frequencies_results))
print("len(theta_frequency_results)", len(theta_frequencies_results))
print("len(alpha_frequency_results)", len(alpha_frequencies_results))
print("len(gamma_frequency_results)", len(gamma_frequencies_results))
print("len(beta_frequency_results)", len(beta_frequencies_results))


print("len(delta_power_results)", len(delta_power_results))
print("len(theta_power_results)", len(theta_power_results))
print("len(alpha_power_results)", len(alpha_power_results))
print("len(gamma_power_results)", len(gamma_power_results))
print("len(beta_power_results)", len(beta_power_results))

# Print first item of each list
print("\nFirst item of delta_frequency_results:", delta_frequencies_results[0])
print("\nFirst item of theta_frequency_results:", theta_frequencies_results[0])
print("\nFirst item of alpha_frequency_results:", alpha_frequencies_results[0])
print("\nFirst item of gamma_frequency_results:", gamma_frequencies_results[0])
print("\nFirst item of beta_frequency_results:", beta_frequencies_results[0])

print("\nFirst item of delta_power_results:", delta_power_results[0])
print("\nFirst item of theta_power_results:", theta_power_results[0])
print("\nFirst item of alpha_power_results:", alpha_power_results[0])
print("\nFirst item of gamma_power_results:", gamma_power_results[0])
print("\nFirst item of beta_power_results:", beta_power_results[0])


def concatenate_lists_by_index(list1, list2, axis=0):
    """
    Concatenate two lists of ndarrays by index along the specified axis.

    Parameters:
    - list1 (list): First list of ndarrays.
    - list2 (list): Second list of ndarrays.
    - axis (int, optional): Axis along which to concatenate. Default is 0.

    Returns:
    - concatenated_list (list): List containing concatenated ndarrays.
    """
    concatenated_list = []
    concatenated_list = [(arr1, arr2) for arr1, arr2 in zip(list1, list2)]
    return concatenated_list

delta_frequency_power_results = concatenate_lists_by_index(delta_frequencies_results, delta_power_results)
theta_frequency_power_results = concatenate_lists_by_index(theta_frequencies_results, theta_power_results)
alpha_frequency_power_results = concatenate_lists_by_index(alpha_frequencies_results, alpha_power_results)
beta_frequency_power_results = concatenate_lists_by_index(beta_frequencies_results, beta_power_results)
gamma_frequency_power_results = concatenate_lists_by_index(gamma_frequencies_results, gamma_power_results)

print("len(delta_frequency_power_results)", len(delta_frequency_power_results))
print("len(theta_frequency_power_results)", len(theta_frequency_power_results))
print("len(alpha_frequency_power_results)", len(alpha_frequency_power_results))
print("len(beta_frequency_power_results)", len(beta_frequency_power_results))
print("len(gamma_frequency_power_results)", len(gamma_frequency_power_results))

# Print first item of each list
print("\nFirst item of delta_frequency_power_results:", delta_frequency_power_results[0])
print("\nFirst item of theta_frequency_power_results:", theta_frequency_power_results[0])
print("\nFirst item of alpha_frequency_power_results:", alpha_frequency_power_results[0])
print("\nFirst item of beta_frequency_power_results:", beta_frequency_power_results[0])
print("\nFirst item of gamma_frequency_power_results:", gamma_frequency_power_results[0])
# In EEG (Electroencephalography), the brain's electrical activity is recorded through electrodes placed on the scalp.
# EEG signals exhibit various rhythmic patterns, known as brainwaves, which correspond to different frequencies.
# The main types of brainwaves commonly observed in EEG recordings are:

# 1. Delta Waves (δ):
#    - Frequency Range: 0.5 - 4 Hz
#    - Associated with deep sleep, unconsciousness, and some abnormal brain activity.

# 2. Theta Waves (θ):
#    - Frequency Range: 4 - 8 Hz
#    - Associated with drowsiness, daydreaming, REM sleep, and certain meditative states.

# 3. Alpha Waves (α):
#    - Frequency Range: 8 - 13 Hz
#    - Predominantly observed during wakeful relaxation with eyes closed, reflecting a state of calmness and relaxation.

# 4. Beta Waves (β):
#    - Frequency Range: 13 - 30 Hz
#    - Associated with active wakefulness, focused attention, cognitive tasks, and mental activity.

# 5. Gamma Waves (γ):
#    - Frequency Range: >30 Hz
#    - Associated with higher cognitive functions, perception, and cross-modal sensory integration.

# # Example PSD data (frequencies and power values)
# frequencies = np.array([0.5, 1, 2, 4, 8, 16, 32])  # Example frequency array
# power_values = np.random.rand(len(frequencies))  # Example power values

# Define frequency ranges for different brainwaves
# Acquire the range of frequencies available in the psd_results data structure





if plot_graphs:
    # Duplicated for sake of simplicity in reading
    frequencies = psd_results[0][0]
    power_values = psd_results[0][1]
    delta_range = (0.5, 4)
    theta_range = (4, 8)
    alpha_range = (8, 13)
    beta_range = (13, 30)
    gamma_range = (30, np.max(frequencies))  # Assuming highest frequency is maximum value in frequencies array

    # Indexing to capture frequency values within desired ranges
    delta_frequencies = frequencies[(frequencies >= delta_range[0]) & (frequencies < delta_range[1])]
    theta_frequencies = frequencies[(frequencies >= theta_range[0]) & (frequencies < theta_range[1])]
    alpha_frequencies = frequencies[(frequencies >= alpha_range[0]) & (frequencies < alpha_range[1])]
    beta_frequencies = frequencies[(frequencies >= beta_range[0]) & (frequencies < beta_range[1])]
    gamma_frequencies = frequencies[(frequencies >= gamma_range[0]) & (frequencies <= gamma_range[1])]

    # Extract corresponding power values
    delta_power = power_values[(frequencies >= delta_range[0]) & (frequencies < delta_range[1])]
    theta_power = power_values[(frequencies >= theta_range[0]) & (frequencies < theta_range[1])]
    alpha_power = power_values[(frequencies >= alpha_range[0]) & (frequencies < alpha_range[1])]
    beta_power = power_values[(frequencies >= beta_range[0]) & (frequencies < beta_range[1])]
    gamma_power = power_values[(frequencies >= gamma_range[0]) & (frequencies <= gamma_range[1])]

    # Print extracted frequency ranges and corresponding power values
    print("Delta Frequencies:", delta_frequencies)
    print("Delta Power:", delta_power)
    print("Theta Frequencies:", theta_frequencies)
    print("Theta Power:", theta_power)
    print("Alpha Frequencies:", alpha_frequencies)
    print("Alpha Power:", alpha_power)
    print("Beta Frequencies:", beta_frequencies)
    print("Beta Power:", beta_power)
    print("Gamma Frequencies:", gamma_frequencies)
    print("Gamma Power:", gamma_power)

    # Create subplots for each frequency band
    plt.figure(figsize=(12, 8))
    # Delta band
    plt.subplot(321)
    plt.plot(delta_frequencies, delta_power, marker='o', linestyle='-')
    plt.title('Theta Band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.ylim(0.1, 1000)  # Set the range of the logarithmic scale (adjust as needed)

    # Theta band
    plt.subplot(322)
    plt.plot(theta_frequencies, theta_power, marker='o', linestyle='-')
    plt.title('Theta Band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.ylim(0.1, 1000)  # Set the range of the logarithmic scale (adjust as needed)

    # Alpha band
    plt.subplot(323)
    plt.plot(alpha_frequencies, alpha_power, marker='o', linestyle='-')
    plt.title('Alpha Band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.ylim(0.1, 1000)  # Set the range of the logarithmic scale (adjust as needed)

    # Beta band
    plt.subplot(324)
    plt.plot(beta_frequencies, beta_power, marker='o', linestyle='-')
    plt.title('Beta Band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.ylim(0.1, 1000)  # Set the range of the logarithmic scale (adjust as needed)

    # Gamma band
    plt.subplot(325)
    plt.plot(gamma_frequencies, gamma_power, marker='o', linestyle='-')
    plt.title('Gamma Band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis scale to logarithmic
    plt.ylim(0.1, 1000)  # Set the range of the logarithmic scale (adjust as needed)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

'''SPECTRAL PEAK FREQUENCY
The spectral peak frequency for each frequency band, you can simply identify the frequency within 
each band where the power is maximized.
'''


# Find spectral peak frequency for each frequency band
def find_peak_frequency(frequencies, power_values):
    # Find index of maximum power value
    peak_index = np.argmax(power_values)
    # Get corresponding frequency
    spectral_peak_freq = frequencies[peak_index]
    return spectral_peak_freq

# Calculate spectral peak frequencies for each frequency band
delta_peak_freq = find_peak_frequency(delta_frequencies, delta_power)
theta_peak_freq = find_peak_frequency(theta_frequencies, theta_power)
alpha_peak_freq = find_peak_frequency(alpha_frequencies, alpha_power)
beta_peak_freq = find_peak_frequency(beta_frequencies, beta_power)
gamma_peak_freq = find_peak_frequency(gamma_frequencies, gamma_power)

# Print spectral peak frequencies for each band
print("\nSpectral peak frequencies:")
print("Delta Peak Frequency:", delta_peak_freq, "Hz")
print("Theta Peak Frequency:", theta_peak_freq, "Hz")
print("Alpha Peak Frequency:", alpha_peak_freq, "Hz")
print("Beta Peak Frequency:", beta_peak_freq, "Hz")
print("Gamma Peak Frequency:", gamma_peak_freq, "Hz")


'''Importing necessary libraries'''

from scipy.stats import entropy
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.signal import argrelextrema
from scipy.stats import gmean


def calculate_entropy(power_values):
    # Normalize power values
    normalized_power = power_values / np.sum(power_values)
    # Calculate entropy
    entropy_value = entropy(normalized_power, base=2)
    return entropy_value

def calculate_spectral_centroid(frequencies, power_values):
    # Calculate spectral centroid
    spectral_centroid = np.sum(frequencies * power_values) / np.sum(power_values)
    return spectral_centroid

def calculate_spectral_flatness(power_values):
    # Calculate geometric mean
    geometric_mean = gmean(power_values)
    # Calculate arithmetic mean
    arithmetic_mean = np.mean(power_values)
    # Calculate spectral flatness
    spectral_flatness = geometric_mean / arithmetic_mean
    return spectral_flatness

def calculate_spectral_edge_frequency(frequencies, power_values, percentile):
    # Calculate cumulative sum of power spectrum
    cumulative_power = np.cumsum(power_values)
    # Normalize cumulative sum
    normalized_cumulative_power = cumulative_power / np.sum(power_values)
    # Find index where cumulative power exceeds the specified percentile
    edge_index = np.argmax(normalized_cumulative_power >= percentile)
    # Get corresponding frequency
    spectral_edge_freq = frequencies[edge_index]
    return spectral_edge_freq

#DELTA
# Compute spectral centroid, spectral flatness, and spectral edge frequency for the delta band
delta_entropy=calculate_entropy(power_values)
delta_centroid = calculate_spectral_centroid(delta_frequencies, delta_power)
delta_flatness = calculate_spectral_flatness(delta_power)
delta_edge_freq_95 = calculate_spectral_edge_frequency(delta_frequencies, delta_power, 0.95)

# Print results
print("\n")
print("Delta Entropy:",delta_entropy)
print("Delta Band Spectral Centroid:", delta_centroid, "Hz")
print("Delta Band Spectral Flatness:", delta_flatness)
print("Delta Band 95% Spectral Edge Frequency:", delta_edge_freq_95, "Hz")


#THETA
# Compute spectral centroid, spectral flatness, and spectral edge frequency for the Theta band
theta_entropy=calculate_entropy(power_values)
theta_centroid = calculate_spectral_centroid(theta_frequencies, theta_power)
theta_flatness = calculate_spectral_flatness(theta_power)
theta_edge_freq_95 = calculate_spectral_edge_frequency(theta_frequencies, theta_power, 0.95)

# Print results
print("\n")
print("Theta Entropy:",theta_entropy)
print("Theta Band Spectral Centroid:", theta_centroid, "Hz")
print("Theta Band Spectral Flatness:", theta_flatness)
print("Theta Band 95% Spectral Edge Frequency:", theta_edge_freq_95, "Hz")



#Alpha
# Compute spectral centroid, spectral flatness, and spectral edge frequency for the Alpha band
alpha_entropy=calculate_entropy(power_values)
alpha_centroid = calculate_spectral_centroid(alpha_frequencies, alpha_power)
alpha_flatness = calculate_spectral_flatness(alpha_power)
alpha_edge_freq_95 = calculate_spectral_edge_frequency(alpha_frequencies, alpha_power, 0.95)

# Print results
print("\n")
print("Alpha Entropy:", alpha_entropy)
print("Alpha Band Spectral Centroid:", alpha_centroid, "Hz")
print("Alpha Band Spectral Flatness:", alpha_flatness)
print("Alpha Band 95% Spectral Edge Frequency:", alpha_edge_freq_95, "Hz")


#Beta
# Compute spectral centroid, spectral flatness, and spectral edge frequency for the Beta band
beta_entropy=calculate_entropy(power_values)
beta_centroid = calculate_spectral_centroid(beta_frequencies, beta_power)
beta_flatness = calculate_spectral_flatness(beta_power)
beta_edge_freq_95 = calculate_spectral_edge_frequency(beta_frequencies, beta_power, 0.95)

# Print results
print("\n")
print("Beta Entropy:", beta_entropy)
print("Beta Band Spectral Centroid:", beta_centroid, "Hz")
print("Beta Band Spectral Flatness:", beta_flatness)
print("Beta Band 95% Spectral Edge Frequency:", beta_edge_freq_95, "Hz")


#GAMMA
# Compute spectral centroid, spectral flatness, and spectral edge frequency for the Beta band
gamma_entropy=calculate_entropy(power_values)
gamma_centroid = calculate_spectral_centroid(gamma_frequencies, gamma_power)
gamma_flatness = calculate_spectral_flatness(gamma_power)
gamma_edge_freq_95 = calculate_spectral_edge_frequency(gamma_frequencies, gamma_power, 0.95)

# Print results
print("\n")
print("Gamma Entropy:", gamma_entropy)
print("Gamma Band Spectral Centroid:", gamma_centroid, "Hz")
print("Gamma Band Spectral Flatness:", gamma_flatness)
print("Gamma Band 95% Spectral Edge Frequency:", gamma_edge_freq_95, "Hz")