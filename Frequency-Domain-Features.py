import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, gmean

class FrequencyDomainFeatures:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fs = 200  # Sampling frequency
        self.psd_results = []
        self.bands = {}
        self.spectral_features = []

    def read_eeg_data(self):
        """Read EEG data from a Parquet file into a DataFrame."""
        self.df_eegs = pd.read_parquet(self.file_path)

    def calculate_psd(self):
        """Calculate the Power Spectral Density (PSD) for each electrode."""
        num_channels = self.df_eegs.shape[1]  # Number of EEG channels
        for i in range(num_channels):
            # Extract EEG signal data from the ith electrode node
            eeg_signal = self.df_eegs.iloc[:, i].values
            # Calculate PSD using Welch's Method
            frequencies, psd = welch(eeg_signal, fs=self.fs, nperseg=self.fs*2)
            # Store Results
            self.psd_results.append((frequencies, psd))
    def extract_frequency_bands(self):
        """Extract frequency bands (delta, theta, alpha, beta, gamma) from PSD."""
        delta_range = (0.5, 4)
        theta_range = (4, 8)
        alpha_range = (8, 13)
        beta_range = (13, 30)
        gamma_range = (30, np.max(self.psd_results[0][0]))

        for frequencies, power_values in self.psd_results:
            bands = {}
            i = 0
            for band_name, band_range in zip(['delta', 'theta', 'alpha', 'beta', 'gamma'],
                                             [delta_range, theta_range, alpha_range, beta_range, gamma_range]):
                band_indices = np.where((frequencies >= band_range[0]) & (frequencies < band_range[1]))[0]
                band_frequencies = frequencies[band_indices]
                band_power_values = power_values[band_indices]
                bands[band_name] = (band_frequencies, band_power_values)
            self.bands[self.df_eegs.columns[i]] = bands
            i += 1

    def calculate_spectral_features(self):
        """Calculate spectral features for each frequency band."""
        spectral_features = []
        for electrode, bands in self.bands.items():
            for band_name, (frequencies, power_values) in bands.items():
                # Calculate spectral centroid
                spectral_centroid = np.sum(frequencies * power_values) / np.sum(power_values)
                # Calculate spectral flatness
                geometric_mean = gmean(power_values)
                arithmetic_mean = np.mean(power_values)
                spectral_flatness = geometric_mean / arithmetic_mean
                # Calculate 95% spectral edge frequency
                cumulative_power = np.cumsum(power_values)
                normalized_cumulative_power = cumulative_power / np.sum(power_values)
                edge_index = np.argmax(normalized_cumulative_power >= 0.95)
                spectral_edge_freq = frequencies[edge_index]
                # Calculate entropy
                normalized_power = power_values / np.sum(power_values)
                entropy_value = entropy(normalized_power, base=2)
                # Store results in a dictionary
                spectral_feature = {
                    "Electrode": electrode,
                    "Band": band_name,
                    "Spectral Centroid": spectral_centroid,
                    "Spectral Flatness": spectral_flatness,
                    "95% Spectral Edge Frequency": spectral_edge_freq,
                    "Entropy": entropy_value
                }
                self.spectral_features.append(spectral_feature)

# Example usage:
if __name__ == "__main__":
    eeg_analysis = FrequencyDomainFeatures("train_eegs/568657.parquet")
    eeg_analysis.read_eeg_data()
    eeg_analysis.calculate_psd()
    eeg_analysis.extract_frequency_bands()
    eeg_analysis.calculate_spectral_features()

    print(eeg_analysis.spectral_features[0])
    print(eeg_analysis.bands)
    print(eeg_analysis.df_eegs)
