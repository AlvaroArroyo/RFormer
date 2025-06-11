import os
import time
import random
from datetime import datetime, timedelta
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import iisignature
import signatory

from model_classification import DecoderTransformer, LSTM_Classification
from datasets import HeartRate
from model_classification import plot_predictions, plot_predictions_signatures

# Reproducibility and device setup
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Transformer on Time series forecasting')
parser.add_argument('--input-size', default=5, type=int, help='input_size (default: 5 = (4 covariates + 1 dim point))')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--eval_batch_size', default=64, type=int, help='eval_batch_size default is equal to training batch_size')
parser.add_argument('--n_head', default=1, type=int, help='n_head (default: 1)')
parser.add_argument('--num-layers', default=1, type=int, help='num-layers (default: 1)')
parser.add_argument('--epoch', default=1000, type=int, help='epoch (default: 20)')
parser.add_argument('--epochs_for_convergence', default=100, type=int, help='number of epochs evaluated to assess convergence (default: 100)')
parser.add_argument('--embedded_dim', default=20, type=int, help='The dimension of position and ID embeddings')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay')
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--overlap', default=False, action='store_true', help='If we overlap prediction range during sampling')
parser.add_argument('--scale_att', default=False, action='store_true', help='Scaling Attention')
parser.add_argument('--sparse', default=False, action='store_true', help='Perform the simulation of sparse attention')
parser.add_argument('--dataset', default='eeg', type=str, help='Dataset you want to train')
parser.add_argument('--v_partition', default=0.1, type=float, help='validation_partition')
parser.add_argument('--q_len', default=1, type=int, help='kernel size for generating key-query')
parser.add_argument('--early_stop_ep', default=60, type=int, help='early_stop_ep')
parser.add_argument('--sub_len', default=1, type=int, help='sub_len of sparse attention')
parser.add_argument('--warmup_proportion', default=-1, type=float, help='warmup_proportion for BERT Adam')
parser.add_argument('--optimizer', default='Adam', type=str, help='Choice BERTAdam or Adam')
parser.add_argument('--continue_training', default=False, action='store_true', help='Load a model and continue training')
parser.add_argument('--save_all_epochs', default=False, action='store_true', help='Save the model at all epochs')
parser.add_argument('--pretrained_model_path', default='', type=str, help='location of a pretrained model')
parser.add_argument('--use_signatures', default=False, action='store_true', help='use dataset signatures')
parser.add_argument('--sig_win_len', default=50, type=int, help='window length for signatures')
parser.add_argument('--sig_level', default=4, type=int, help='signature level')
parser.add_argument('--hyperp_tuning', default=False, action='store_true', help='perform hyperparameter tuning')
parser.add_argument('--downsampling', default=False, action='store_true', help='undersample the signal')
parser.add_argument('--zero_shot_downsample', default=False, action='store_true', help='zero-shot downsampling')
parser.add_argument('--model', default='transformer', type=str, help='Model to train')
parser.add_argument('--overlapping_sigs', default=False, action='store_true', help='use overlapping signatures')
parser.add_argument('--univariate', default=False, action='store_true', help='use univariate signatures')
parser.add_argument('--stack', default=False, action='store_true', help='use multi-view attention')
parser.add_argument('--irreg', default=True, action='store_true', help='make inputs time-invariant')
parser.add_argument('--num_windows', default=75, type=int, help='number of windows')
parser.add_argument('--random', default=False, action='store_true', help='drop random elements')
parser.add_argument('--preprocess', default=False, action='store_true', help='preprocess data using signatures')
args = parser.parse_args()
print(args)

# Signature utility functions

def Signature_overlapping_univariate(data, depth, win_len, device):
    sigs = [iisignature.sig(data.cpu()[:, :, i].unsqueeze(2), depth, 2) for i in range(data.shape[2])]
    sigs = np.concatenate(sigs, axis=2)
    indices = np.arange(win_len - 2, data.shape[1], win_len)
    return torch.tensor(sigs)[:, indices, :].to(device).float()


def Signature_nonoverlapping_univariate(data, depth, win_len, device):
    B, T, F = data.shape
    n_windows = T // win_len
    reshaped = data[:, :n_windows * win_len, :].reshape(B, n_windows, win_len, F).cpu()
    sigs = [iisignature.sig(reshaped[:, :, :, i], depth) for i in range(F)]
    sigs = np.concatenate(sigs, axis=2)
    return torch.tensor(sigs).to(device).float()


def Signature_overlapping(data, depth, win_len, device):
    sigs = iisignature.sig(data.cpu(), depth, 2)
    indices = np.arange(win_len - 2, data.shape[1], win_len)
    return torch.tensor(sigs)[:, indices, :].to(device).float()


def Signature_overlapping_irreg(data, depth, num_windows, x, device):
    sigs = iisignature.sig(data.cpu(), depth, 2)
    steps = np.linspace(0, x.max(), num_windows + 1)[1:]
    indices = [int(np.searchsorted(x, s)) - 1 for s in steps]
    return torch.tensor(sigs)[:, indices, :].to(device).float()


def Signature_nonoverlapping(data, depth, win_len, device):
    B, T, F = data.shape
    n_windows = T // win_len
    reshaped = data.reshape(B, n_windows, win_len, F).cpu()
    sigs = iisignature.sig(reshaped, depth)
    return torch.tensor(sigs).to(device).float()


def Signature_nonoverlapping_irreg(data, depth, num_windows, x, device):
    step = x.max() / num_windows
    boundaries = [0] + [int(np.searchsorted(x, step * i)) for i in range(1, num_windows + 1)]
    data_cpu = data.cpu()
    sigs_list = []
    for i in range(len(boundaries) - 1):
        seg = data_cpu[:, boundaries[i]:boundaries[i+1], :]
        sig = iisignature.sig(seg, depth).reshape(data.shape[0], 1, -1)
        sigs_list.append(sig)
    return torch.tensor(np.concatenate(sigs_list, axis=1)).to(device).float()


def apply_signatures(inputs, args, x, device):
    # Choose appropriate signature transformation
    if not args.stack:
        if args.overlapping_sigs and args.univariate:
            return Signature_overlapping_univariate(inputs, args.sig_level, args.sig_win_len, device)
        if args.overlapping_sigs and not args.univariate:
            if args.irreg:
                return Signature_overlapping_irreg(inputs, args.sig_level, args.num_windows, x, device)
            return Signature_overlapping(inputs, args.sig_level, args.sig_win_len, device)
        if not args.overlapping_sigs and args.univariate:
            return Signature_nonoverlapping_univariate(inputs, args.sig_level, args.sig_win_len, device)
        if not args.overlapping_sigs and not args.univariate:
            if args.irreg:
                return Signature_nonoverlapping_irreg(inputs, args.sig_level, args.num_windows, x, device)
            return Signature_nonoverlapping(inputs, args.sig_level, args.sig_win_len, device)
    else:
        # Multi-view: concatenate overlapping and non-overlapping
        if not args.univariate:
            if args.irreg:
                nonov = Signature_nonoverlapping_irreg(inputs, args.sig_level, args.num_windows, x, device)
                ov = Signature_overlapping_irreg(inputs, args.sig_level, args.num_windows, x, device)
            else:
                nonov = Signature_nonoverlapping(inputs, args.sig_level, args.sig_win_len, device)
                ov = Signature_overlapping(inputs, args.sig_level, args.sig_win_len, device)
        else:
            nonov = Signature_nonoverlapping_univariate(inputs, args.sig_level, args.sig_win_len, device)
            ov = Signature_overlapping_univariate(inputs, args.sig_level, args.sig_win_len, device)
        return torch.cat((nonov, ov), dim=2)


def calculate_MSE(args, model, data_loader, dataset_size):
    model.eval()
    total_loss = 0.0
    mse_loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.zero_shot_downsample:
                if args.random:
                    keep = sorted(random.sample(range(inputs.shape[1]), len(range(inputs.shape[1]))))
                x = x[keep]
                inputs = inputs[:, keep, :]
            if args.use_signatures and not args.preprocess:
                inputs = apply_signatures(inputs, args, x, device)
            outputs = model(inputs).squeeze(-1)
            total_loss += mse_loss(outputs, labels)
    rmse = torch.sqrt(total_loss / dataset_size)
    return rmse.item()


def main():
    print('Using signatures') if args.use_signatures else print('Not using signatures')

    # Data loading
    if args.eval_batch_size < 1:
        eval_bs = args.batch_size
    else:
        eval_bs = args.eval_batch_size

    if args.dataset != 'eeg':
        raise ValueError('Dataset not supported')

    hr = HeartRate()
    train_ds, test_ds, valid_ds = hr.get_heart_rate()
    seq_len_orig = train_ds[0][0].shape[0]
    num_features = train_ds[0][0].shape[1]
    num_samples = len(train_ds)
    num_classes = 1

    # Data loaders
    if args.preprocess:
        full_loader = lambda ds: DataLoader(ds, batch_size=len(ds), shuffle=(ds is train_ds))
        train_loader = full_loader(train_ds)
        val_loader = full_loader(valid_ds)
        test_loader = full_loader(test_ds)

        # Precompute signatures
        for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
            t0 = time.time()
            batch = next(iter(loader))
            inputs, labels = batch
            inputs = apply_signatures(inputs, args, np.linspace(0, inputs.shape[1], inputs.shape[1]), device)
            print(f"{name} sigs computed in {time.time() - t0:.2f}s")
        # Rebuild datasets
        train_ds = TensorDataset(inputs, labels)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valid_ds, batch_size=eval_bs, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False)

    # Feature and sequence adjustments for signatures
    seq_len = seq_len_orig
    if args.use_signatures:
        base_dim = signatory.signature_channels(num_features, args.sig_level) if not args.univariate else num_features * args.sig_level
        num_features = base_dim * (2 if args.stack else 1)
        seq_len = args.num_windows if args.irreg else seq_len_orig // args.sig_win_len
        print('Num features:', num_features, 'Seq len:', seq_len)

    # Prepare downsampling indices
    all_indices = list(range(seq_len_orig))
    if args.random:
        indices_keep = sorted(random.sample(all_indices, 100))
    else:
        indices_keep = all_indices[::2] if args.downsampling else all_indices

    # Model setup
    if args.model == 'transformer':
        model = DecoderTransformer(args, input_dim=num_features, n_head=args.n_head,
                                   layer=args.num_layers, seq_num=num_samples,
                                   n_embd=args.embedded_dim, win_len=seq_len,
                                   num_classes=num_classes).to(device)
    else:
        model = LSTM_Classification(input_size=num_features, hidden_size=10,
                                    num_layers=100, batch_first=True,
                                    num_classes=num_classes).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.random:
                indices_keep = sorted(random.sample(all_indices, len(indices_keep)))
                inputs = inputs[:, indices_keep, :]
                x = x[indices_keep]
            if args.use_signatures and not args.preprocess:
                inputs = apply_signatures(inputs, args, x, device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        val_rmse = calculate_MSE(args, model, test_loader, len(test_ds))
        print(f"Epoch {epoch+1}/{args.epoch} - Train Loss: {avg_loss:.4f}  Val RMSE: {val_rmse:.4f}")

        # Early stopping and LR adjustment
        if val_rmse < best_val:
            best_val = val_rmse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > 15:
                args.lr /= 10
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr
                print("Reduced LR to", args.lr)
        if epoch - (epoch - epochs_no_improve) >= args.early_stop_ep:
            print("Early stopping at epoch", epoch)
            break

    # Final evaluation
    test_rmse = calculate_MSE(args, model, test_loader, len(test_ds))
    print(f"Test RMSE: {test_rmse:.4f}")
    plot_fn = plot_predictions_signatures if args.use_signatures else plot_predictions
    plot_fn(device, args, model, test_loader, './out', num_predictions=2 if args.use_signatures else 10)

if __name__ == '__main__':
    main()
