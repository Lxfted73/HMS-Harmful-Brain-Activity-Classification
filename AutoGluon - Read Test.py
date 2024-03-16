import pandas as pd
from autogluon.tabular import FeatureMetadata
from autogluon.tabular import TabularDataset, TabularPredictor

test_csv = pd.read_csv('test.csv')
# print(test_csv.head())

test_eeg_df = pd.read_parquet('test_eegs/3911565283.parquet')
# print(test_eeg_df.head())

test_spectrograms_df = pd.read_parquet('test_spectrograms/853520.parquet')
# print(test_eeg_df)

patient_profile = {}

for index, patient_data in test_csv.itertuples():
    print(patient_data.spectrogram_id)
    patient_profile['eeg'] = pd.read_parquet(f'test_eegs/{patient_data.eeg_id}.parquet')
    patient_profile['spectrogram'] = pd.read_parquet(f'test_spectrograms/{patient_data.spectrogram_id}.parquet')


