"""
The dataset preprocessing is based on DeepAR
https://arxiv.org/pdf/1704.04110.pdf
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EthanolLevel(Dataset):
    #https://www.timeseriesclassification.com/description.php?Dataset=EthanolLevel
    def __init__(self, features, labels):
        #self.features = np.expand_dims(features, axis=-1)
        self.features = np.transpose(features, (0, 2, 1))
        self.labels = labels.astype(np.int64) #Originally, the labels are 1,2,3,4, in the next line we make them 0,1,2,3 to avoid problems with the BCElss
        self.labels = self.labels - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
    # Ensure the signal data is formatted as expected by the model, which might expect input size of [sequence_length, num_features]
        signal = self.features[index]
        label = self.labels[index]
        sample = {'input': torch.tensor(signal, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}
        return sample



class EthanolConcentration(Dataset):
    #https://www.timeseriesclassification.com/description.php?Dataset=EthanolLevel
    def __init__(self, features, labels):
        #self.features = np.expand_dims(features, axis=-1)
        self.features = np.transpose(features, (0, 2, 1))
        _, self.labels = np.unique(labels, return_inverse=True)
        # self.labels = labels.astype(np.int64) #Originally, the labels are 1,2,3,4, in the next line we make them 0,1,2,3 to avoid problems with the BCElss
        # self.labels = self.labels - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
    # Ensure the signal data is formatted as expected by the model, which might expect input size of [sequence_length, num_features]
        signal = self.features[index]
        label = self.labels[index]
        sample = {'input': torch.tensor(signal, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}
        return sample

class TimeSeriesClassification(Dataset):
    # format for generic dataset from https://www.timeseriesclassification.com
    def __init__(self, features, labels):
        self.features = np.transpose(features, (0, 2, 1))
        _, self.labels = np.unique(labels, return_inverse=True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        signal = self.features[index]
        label = self.labels[index]
        sample = {'input': torch.tensor(signal, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}
        return sample 

class TimeSeriesClassification_preprocess(Dataset):
    # format for generic dataset from https://www.timeseriesclassification.com
    def __init__(self, features, labels):
        self.features = features
        _, self.labels = np.unique(labels, return_inverse=True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        signal = self.features[index]
        label = self.labels[index]
        sample = {'input': signal, 'label': torch.tensor(label, dtype=torch.long)}
        return sample    





