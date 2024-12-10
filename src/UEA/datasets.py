"""
The dataset preprocessing is based on DeepAR
https://arxiv.org/pdf/1704.04110.pdf
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sktime.datasets import load_from_arff_to_dataframe
from torch import Tensor
import os, os.path
import urllib.response
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import os
import urllib.request
import tarfile
import shutil
import librosa
import torch.utils.data as data
from scipy import integrate
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


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





