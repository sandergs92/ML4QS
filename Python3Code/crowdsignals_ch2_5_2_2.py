# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Chapter 2.5.2.2

RESULT_PATH = Path('./intermediate_datafiles/chapter2_result.csv')
EXPERIMENT_PATH = Path('./intermediate_datafiles/chapter2_result.csv')

# Create an initial dataset object
dataset = pd.read_csv(Path(RESULT_PATH), index_col=0)
dataset.index = pd.to_datetime(dataset.index)
dataset_exp = pd.read_csv(Path(EXPERIMENT_PATH), index_col=0)
dataset_exp.index = pd.to_datetime(dataset_exp.index)

# Kolmogorov–Smirnov test
crowd_acc_phone_z = dataset.loc[dataset['labelWalking'] == 1]['acc_phone_z']
experiment_acc_phone_z = dataset_exp.loc[dataset_exp['labelWalking'] == 1]['acc_phone_z']
if ks_2samp(crowd_acc_phone_z, experiment_acc_phone_z)[1] < 0.05:
    print(ks_2samp(crowd_acc_phone_z, experiment_acc_phone_z))
    print('We have sufficient evidence to say that the two sample datasets do not come from the same distribution.')
else: 
    print(ks_2samp(crowd_acc_phone_z, experiment_acc_phone_z))
    print('We have sufficient evidence to say that the two sample datasets do come from the same distribution.')

# Normalize the data before plotting
cols = [
    'acc_phone_x', 'acc_phone_y', 'acc_phone_z', 
    'gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z', 
    'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 
    ]
for col in cols:
    col_min = dataset[col].min()
    col_max = dataset[col].max()
    col_range = dataset[col].count()
    # Ringo's formula
    # dataset[col] = [((x - col_min) / col_range) for x in dataset[col]]
    # Formula to normalize between 0 and 1
    dataset[col] = [((x - col_min) / (col_max - col_min)) for x in dataset[col]]

# Plot the data
DataViz = VisualizeDataset(__file__)

# Plot all data
DataViz.plot_dataset(dataset, ['acc_phone_', 'gyr_phone_', 'mag_phone_', 'label'],
                                ['like', 'like', 'like', 'like'],
                                ['line', 'line', 'line', 'points'])

# Lastly, print a statement to know the code went through
print('The code has run through successfully!')