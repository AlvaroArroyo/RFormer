
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.distributions.normal import Normal
import copy
from torch.nn.parameter import Parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *



class LSTM_Classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first)
        self.linear = torch.nn.Linear(hidden_size, num_classes, bias=True)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.mean(dim=1) 
        x = self.linear(x)
        return x