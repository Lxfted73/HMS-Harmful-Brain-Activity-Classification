import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


# Read Parquet file into a DataFrame
df_test_eegs = pd.read_parquet("Data/test_eegs.parquet")

#Check for Missing Values
missing_values = df_test_eegs.isnull().sum()
print("Missing Values:\n", missing_values)

#no missing values

#Calculate Mean Value
mean_values = df_test_eegs.mean()
print("Mean Values:\n", mean_values)


# Calculate Min and Max Values
min_values = df_test_eegs.min()
max_values = df_test_eegs.max()
print("Min Values:\n", min_values)
print("Max Values:\n", max_values)

pd.set_option('display.max_columns', None)


# Calculate Standard Deviation, Variance, and RMS Values
std_values = df_test_eegs.std()
variance_values = df_test_eegs.var()
rms_values = np.sqrt(np.mean(df_test_eegs**2))

print("Standard Deviation:\n", std_values)
print("Variance:\n", variance_values)
print("RMS:\n", rms_values)

pd.set_option('display.max_columns', None)

#2
df_test_spectogram=pd.read_parquet("test_spectogram.parquet")
mean_values=df_test_spectogram.mean()
print("Mean Values:\n", mean_values)

min_values = df_test_spectogram.min()
max_values = df_test_spectogram.max()
print("Min Values:\n", min_values)
print("\nMax Values:\n", max_values)

std_values = df_test_spectogram.std()
variance_values = df_test_spectogram.var()
rms_values = np.sqrt(np.mean(df_test_spectogram**2))

print("Standard Deviation:\n", std_values)
print("\n\nVariance:\n", variance_values)
print("\n\nRMS:\n", rms_values)


#3
df_train_eegs=pd.read_parquet("train_eegs.parquet")

mean_values = df_train_eegs.mean()
print("Mean Values:\n", mean_values)

min_values = df_train_eegs.min()
max_values = df_train_eegs.max()
print("Min Values:\n", min_values)
print("\nMax Values:\n", max_values)

std_values = df_train_eegs.std()
variance_values = df_train_eegs.var()
rms_values = np.sqrt(np.mean(df_train_eegs**2))

print("Standard Deviation:\n", std_values)
print("\n\nVariance:\n", variance_values)
print("\n\nRMS:\n", rms_values)


#4
df_train_spectogram=pd.read_parquet("train_spectogram.parquet")
mean_values = df_train_spectogram.mean()
print("Mean Values:\n", mean_values)

min_values = df_train_spectogram.min()
max_values = df_train_spectogram.max()
print("Min Values:\n", min_values)
print("\nMax Values:\n", max_values)

std_values = df_train_spectogram.std()
variance_values = df_train_spectogram.var()
rms_values = np.sqrt(np.mean(df_train_spectogram**2))

print("Standard Deviation:\n", std_values)
print("\n\nVariance:\n", variance_values)
print("\n\nRMS:\n", rms_values)






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

