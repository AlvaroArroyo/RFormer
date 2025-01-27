import time
import math
import numpy as np
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader
from datasets import *
import iisignature
import wandb

from sig_utils import ComputeSignatures



def ComputeModelParams(seq_length_original, num_features_original, config):
    if config.use_signatures:
        
        num_channels = num_features_original
        if config.add_time:
            num_channels += 1

        seq_length = config.num_windows
        num_features = 0
        if config.univariate:
            if config.global_backward:
                num_features += num_features_original*iisignature.siglength(2, config.sig_level)
            if config.global_forward:
                num_features += num_features_original*iisignature.siglength(2, config.sig_level)
            if config.local_tight:
                num_features += num_features_original*iisignature.siglength(2, config.sig_level)
            if config.local_wide:
                num_features += num_features_original*iisignature.siglength(2, config.sig_level)
        else:
            if config.global_backward:
                num_features += iisignature.siglength(num_channels, config.sig_level)
            if config.global_forward:
                num_features += iisignature.siglength(num_channels, config.sig_level)
            if config.local_tight:
                num_features += iisignature.siglength(num_channels, config.sig_level)
            if config.local_wide:
                num_features += iisignature.siglength(num_channels, config.sig_level)

        compression = (seq_length*num_features)/(seq_length_original*num_features_original)
        compression_L2d = (seq_length*seq_length*num_features)/(seq_length_original*seq_length_original*num_features_original)
        print('Num features without signature: ', num_features_original, 'Num features with signatures: ', num_features)
        print('Sequence length without signature: ', seq_length_original, 'Sequence length with signatures: ', seq_length)
        print()
        print(f"Compression: {compression}")
        print()
        print(f"Compression L^2d: {compression_L2d}")
        print()
        if config.wandb:
            wandb.log({
                "metrics/compression":compression,
                "metrics/compression_L2d":compression_L2d
            })


    else:
        seq_length = seq_length_original
        num_features = num_features_original
        compression = 1.0
        print(f"Number of features: {num_features}", f'Sequence length: {seq_length}')
        print()

    return num_features, seq_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def leadlag(X):
    '''
    Returns lead-lag-transformed stream of X

    Arguments:
        X: list, whose elements are tuples of the form
        (time, value).

    Returns:
        list of points on the plane, the lead-lag
        transformed stream of X
    '''

    l=[]

    for j in range(2*(len(X))-1):
        i1=j//2
        i2=j//2
        if j%2!=0:
            i1+=1
        l.append(np.concatenate([X.loc[i1].values[:], X.loc[i2].values[:]]))

    return np.stack(l)

def save_checkpoint(state, epoch , loss_is_best, filename, save_all_epochs):
    if(save_all_epochs):
        torch.save(state, filename+'/'+str(epoch)+'_v_loss.pth.tar')
    if(loss_is_best):
        torch.save(state, filename+'/best_v_loss.pth.tar')


def get_dataset_preprocess(config, seed, device):
    # Create dataset and data loader
    if(config.eval_batch_size ==-1):
        eval_batch_size = config.batch_size
    else:
        eval_batch_size = config.eval_batch_size

    if config.dataset[:4] == "TSC_":
        ds_key = config.dataset[4:]

        X, Y = load_classification(ds_key)

        X = torch.tensor(np.transpose(X, (0, 2, 1)))
        seq_length_original = X.shape[1]

        ComputeModelParams(X.shape[1], X.shape[2], config)

        indices_keep = [i for i in range(seq_length_original)]
        x = np.linspace(0, X.shape[1], X.shape[1])
        x = x[indices_keep]
        if config.add_time:
            t = (torch.linspace(0, seq_length_original, seq_length_original)/seq_length_original).reshape(-1,1)
            X = torch.cat([t.repeat(X.shape[0], 1, 1), X], dim=2)
        X = ComputeSignatures(X, x, config, device)
        
        seq_length_original = X.shape[1]
        num_classes = len(np.unique(Y))
        num_features = X.shape[2]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=config.test_size, random_state=seed)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=config.val_size, random_state=seed)
        
        train_dataset = TimeSeriesClassification_preprocess(x_train, y_train)
        val_dataset = TimeSeriesClassification_preprocess(x_val, y_val)
        test_dataset = TimeSeriesClassification_preprocess(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        num_samples = len(x_train)
    else:
        raise NotImplementedError('Just TSC for now')

    return train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features


def get_dataset(config, seed):
    # Create dataset and data loader
    if(config.eval_batch_size ==-1):
        eval_batch_size = config.batch_size
    else:
        eval_batch_size = config.eval_batch_size

    if config.dataset[:4] == "TSC_":
        ds_key = config.dataset[4:]

        X, Y = load_classification(ds_key)
        seq_length_original = X.shape[-1]
        num_classes = len(np.unique(Y))

        num_features = X.shape[1]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=config.test_size, random_state=seed)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=config.val_size, random_state=seed)
        
        train_dataset = TimeSeriesClassification(x_train, y_train)
        val_dataset = TimeSeriesClassification(x_val, y_val)
        test_dataset = TimeSeriesClassification(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        num_samples = len(x_train)
    else:
        raise NotImplementedError('Just TSC for now')

    return train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features

def signature_channels(channels: int, depth: int, scalar_term: bool = False) -> int:
    """
    Computes the number of output channels for a signature call.

    Args:
        channels (int): Number of input channels.
        depth (int): Depth of the signature computation.
        scalar_term (bool): Whether to include the constant scalar term.

    Returns:
        int: Total number of output channels.
    """
    result = sum(channels**d for d in range(1, depth + 1))
    if scalar_term:
        result += 1
    return result