import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Read Parquet file into a DataFrame
df_test_eegs = pd.read_parquet("Data/test_eegs.parquet")
# df_test_spectograms = pd.read_parquet("/Data/test_spectogram.parquet")
# df_train_eegs = pd.read_parquet("Data/train_eegs.parquet")
# df_train_spectograms = pd.read_parquet("Data/train_spectogram.parquet")

pd.set_option('display.max_columns', None)

# print("df_train_eegs.shape", df_test_eegs.shape)
# print("df_train_eegs.head()\n", df_test_eegs.head)

electrodes = df_test_eegs.columns[:-1]  # Exclude the last column (EKG)
electrode_voltages = df_test_eegs[electrodes]
print("Electrodes: \n", electrodes,"Electrode Locations: \n", electrode_voltages)

# Extract signal amplitude features
signal_amplitudes_mean = df_test_eegs[electrodes].mean(axis=0)
signal_amplitudes_max = df_test_eegs[electrodes].max(axis=0)
signal_amplitudes_min = df_test_eegs[electrodes].min(axis=0)
signal_amplitudes_variance = df_test_eegs[electrodes].var(axis=0)

df_test_eegs['Mean_Amplitude'] = signal_amplitudes_mean.values
df_test_eegs['Max_Amplitude'] = signal_amplitudes_max.values
df_test_eegs['Min_Amplitude'] = signal_amplitudes_min.values
df_test_eegs['Variance_Amplitude'] = signal_amplitudes_variance.values

# Print the updated DataFrame
print(df_test_eegs)

# # Print the calculated features
# print("Mean Signal Amplitudes:")
# print(signal_amplitudes_mean)
# print("\nMaximum Signal Amplitudes:")
# print(signal_amplitudes_max)
# print("\nMinimum Signal Amplitudes:")
# print(signal_amplitudes_min)
# print("\nSignal Amplitudes Variance:")
# print(signal_amplitudes_variance)

