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

#Print Data Shape
print("df_test_eegs.shape:", df_test_eegs.shape)
print("df_test_eegs.head():\n", df_test_eegs.head())

# Plot one electrode of data:

# Sampling frequency
sampling_frequency = 200  # samples per second
# Number of samples
num_samples = 10000  # for example, plot 10000 samples
# Time per sample
time_per_sample = 1 / sampling_frequency  # seconds per sample
# Create an array of sample indices
sample_indices = np.arange(num_samples)
# Convert sample indices to time in seconds
time_seconds = sample_indices * time_per_sample

plt.figure(figsize=(10,6))

# Add title and labels
plt.plot(time_seconds,df_test_eegs['Fp1'])
plt.title("Frequency of Fp1")
plt.xlabel('Seconds')
plt.ylabel('Voltage (units)')

# Show plot
plt.show()

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
df_test_spectogram=pd.read_parquet("Data/test_spectogram.parquet")
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
df_train_eegs=pd.read_parquet("Data/train_eegs.parquet")

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
df_train_spectogram=pd.read_parquet("Data/train_spectogram.parquet")
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




