import pandas as pd
from autogluon.tabular import FeatureMetadata
from autogluon.tabular import TabularDataset, TabularPredictor

train_csv = pd.read_csv('train.csv')
# print(train_csv.head())

train_eeg_df = pd.read_parquet('train_eegs/3911565283.parquet')
# print(train_eeg_df.head())

train_spectrograms_df = pd.read_parquet('train_spectrograms/853520.parquet')
# print(train_eeg_df)

patient_profile = {}

for index, patient_data in train_csv.itertuples():
    print(patient_data.spectrogram_id)
    patient_profile['eeg'] = pd.read_parquet(f'train_eegs/{patient_data.eeg_id}.parquet')
    patient_profile['spectrogram'] = pd.read_parquet(f'train_spectrograms/{patient_data.spectrogram_id}.parquet')


