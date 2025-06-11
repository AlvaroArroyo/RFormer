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



class SinusoidalDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)
        
    
        # Add more noise to the signal
        #signal += 0.1 * np.random.randn(self.seq_length)

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    
class SinusoidalDatasetLong(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)

        # Add more noise to the signal
        signal += 0.1 * np.random.randn(self.seq_length)

        signal2 = 0.5 * np.sin(2 * np.pi * (freq / 5) * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)
        signal2 += 0.2 * np.sin(4 * np.pi * (freq / 5) * t) + 0.1 * np.sin(8 * np.pi * (freq / 5) * t)
        signal2 += 0.1 * np.random.randn(self.seq_length)

        signal[250:] = signal2[250:]

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    

