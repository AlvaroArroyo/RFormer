import time
import math
import numpy as np
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader
from datasets import *
import iisignature

def apply_signatures(data, args, x, device):
    """Apply signature transforms based on args settings."""
    if not args.stack:
        # Single view
        if args.overlapping_sigs:
            if args.univariate:
                return Signature_overlapping_univariate(data, args.sig_level, args.sig_win_len, device)
            return (Signature_overlapping_irreg(data, args.sig_level, args.num_windows, x, device)
                    if args.irreg else Signature_overlapping(data, args.sig_level, args.sig_win_len, device))
        else:
            if args.univariate:
                return Signature_nonoverlapping_univariate(data, args.sig_level, args.sig_win_len, device)
            return (Signature_nonoverlapping_irreg(data, args.sig_level, args.num_windows, x, device)
                    if args.irreg else Signature_nonoverlapping(data, args.sig_level, args.sig_win_len, device))
    else:
        # Stacked: concatenate overlapping and non-overlapping views
        if args.univariate:
            s1 = Signature_nonoverlapping_univariate(data, args.sig_level, args.sig_win_len, device)
            s2 = Signature_overlapping_univariate(data, args.sig_level, args.sig_win_len, device)
        else:
            if args.irreg:
                s1 = Signature_nonoverlapping_irreg(data, args.sig_level, args.num_windows, x, device)
                s2 = Signature_overlapping_irreg(data, args.sig_level, args.num_windows, x, device)
            else:
                s1 = Signature_nonoverlapping(data, args.sig_level, args.sig_win_len, device)
                s2 = Signature_overlapping(data, args.sig_level, args.sig_win_len, device)
        return torch.cat((s1, s2), dim=2)

# Signature functions

def Signature_overlapping_univariate(data, depth, win_len, device):
    """Compute overlapping signatures on each channel separately."""
    # data: Tensor[B, T, F]
    sigs = [
        iisignature.sig(
            data.cpu()[:, :, i].unsqueeze(2), depth, 2
        ) for i in range(data.shape[2])
    ]
    sigs = np.concatenate(sigs, axis=2)
    indices = np.arange(win_len - 2, data.shape[1], win_len)
    return torch.tensor(sigs)[:, indices, :].to(device).float()


def Signature_nonoverlapping_univariate(data, depth, win_len, device):
    """Compute non-overlapping signatures on each channel separately."""
    B, T, F = data.shape
    n_windows = T // win_len
    data_ = data[:, :n_windows * win_len, :].reshape(B, n_windows, win_len, F).cpu()
    sigs = [
        iisignature.sig(data_[:, :, :, i], depth)
        for i in range(F)
    ]
    sigs = np.concatenate(sigs, axis=2)
    return torch.tensor(sigs).to(device).float()


def Signature_overlapping(data, depth, win_len, device):
    """Compute overlapping signatures on full multivariate path."""
    sigs = iisignature.sig(data.cpu(), depth, 2)
    indices = np.arange(win_len - 2, data.shape[1], win_len)
    return torch.tensor(sigs)[:, indices, :].to(device).float()


def Signature_overlapping_irreg(data, depth, num_windows, x, device):
    """Compute overlapping signatures on irregularly sampled data."""
    sigs = iisignature.sig(data.cpu(), depth, 2)
    steps = np.linspace(0, x.max(), num_windows + 1)[1:]
    indices = [int(np.searchsorted(x, s)) - 1 for s in steps]
    return torch.tensor(sigs)[:, indices, :].to(device).float()


def Signature_nonoverlapping(data, depth, win_len, device):
    """Compute non-overlapping signatures on full multivariate path."""
    B, T, F = data.shape
    n_windows = T // win_len
    reshaped = data.reshape(B, n_windows, win_len, F).cpu()
    sigs = iisignature.sig(reshaped, depth)
    return torch.tensor(sigs).to(device).float()


def Signature_nonoverlapping_irreg(data, depth, num_windows, x, device):
    """Compute non-overlapping signatures on irregularly sampled data."""
    step = x.max() / num_windows
    boundaries = [0] + [int(np.searchsorted(x, step * i)) for i in range(1, num_windows + 1)]
    data_cpu = data.cpu()
    sigs_list = []
    for i in range(len(boundaries) - 1):
        seg = data_cpu[:, boundaries[i]:boundaries[i+1], :]
        sig = iisignature.sig(seg, depth).reshape(data.shape[0], 1, -1)
        sigs_list.append(sig)
    return torch.tensor(np.concatenate(sigs_list, axis=1)).to(device).float()


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
