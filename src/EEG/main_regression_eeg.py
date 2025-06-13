import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse

from datetime import datetime, timedelta
import os
import statistics

import iisignature
import pandas as pd
import signatory

from datasets import *
from sig_utils import *
from utils import *
from model_classification import *


from torch.utils.data import Dataset
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)


parser = argparse.ArgumentParser(description='PyTorch Transformer on Time series forecasting')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--eval_batch_size', default=64, type=int, help='eval_batch_size default is equal to training batch_size')
parser.add_argument('--n_head', default=1, type=int, help='n_head (default: 1)')
parser.add_argument('--num-layers', default=1, type=int, help='num-layers (default: 1)')
parser.add_argument('--epoch', default=1000, type=int, help='epoch (default: 20)')
parser.add_argument('--embedded_dim', default=20, type=int, help='The dimension of position and ID embeddings')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--scale_att', default=False, action='store_true', help='Scaling Attention')
parser.add_argument('--sparse', default=False, action='store_true', help='Perform the simulation of sparse attention')
parser.add_argument('--dataset', default='eeg', type=str, help='Dataset you want to train')
parser.add_argument('--v_partition', default=0.1, type=float, help='validation_partition')
parser.add_argument('--q_len', default=1, type=int, help='kernel size for generating key-query')
parser.add_argument('--early_stop_ep', default=60, type=int, help='early_stop_ep')
parser.add_argument('--sub_len', default=1, type=int, help='sub_len of sparse attention')
parser.add_argument('--use_signatures', default=False, action='store_true', help='use dataset signatures')
parser.add_argument('--sig_win_len', default=50, type=int, help='window length for signatures')
parser.add_argument('--sig_level', default=4, type=int, help='signature level')
parser.add_argument('--zero_shot_downsample', default=False, action='store_true', help='zero-shot downsampling')
parser.add_argument('--model', default='transformer', type=str, help='Model to train')
parser.add_argument('--overlapping_sigs', default=False, action='store_true', help='use overlapping signatures')
parser.add_argument('--univariate', default=False, action='store_true', help='use univariate signatures')
parser.add_argument('--stack', default=False, action='store_true', help='use multi-view attention')
parser.add_argument('--irreg', default=True, action='store_true', help='make inputs time-invariant')
parser.add_argument('--num_windows', default=75, type=int, help='number of windows')
parser.add_argument('--random', default=False, action='store_true', help='drop random elements')
parser.add_argument('--preprocess', default=False, action='store_true', help='preprocess data using signatures')
    

global args
args = parser.parse_args()
print(args)

def sigs(inputs):
    x = np.linspace(0, inputs.shape[1], inputs.shape[1])

    if not args.stack:
        if args.overlapping_sigs and args.univariate:
            inputs=Signature_overlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
        elif args.overlapping_sigs and not args.univariate:
            if args.irreg:
                inputs = Signature_overlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
            else:
                inputs = Signature_overlapping(inputs, args.sig_level, args.sig_win_len, device)
        elif not args.overlapping_sigs and args.univariate:
            inputs = Signature_nonoverlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
        elif not args.overlapping_sigs and not args.univariate:
            if args.irreg:
                inputs = Signature_nonoverlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
            else:
                inputs = Signature_nonoverlapping(inputs,args.sig_level, args.sig_win_len, device)
    else:
        if not args.univariate:
            if args.irreg:
                inputs1 = Signature_nonoverlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
                inputs2 = Signature_overlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
            else:
                inputs1 = Signature_nonoverlapping(inputs,args.sig_level, args.sig_win_len, device)
                inputs2 = Signature_overlapping(inputs,args.sig_level, args.sig_win_len, device)
            inputs = torch.cat((inputs1, inputs2), dim=2)
            
        else:
            inputs1 = Signature_nonoverlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
            inputs2 = Signature_overlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
            inputs = torch.cat((inputs1, inputs2), dim=2)

    return inputs


def get_eval_batch_size(args):
    return args.batch_size if args.eval_batch_size == -1 else args.eval_batch_size


def load_heart_rate_data(args):
    dataset = HeartRate()
    train_ds, test_ds, valid_ds = dataset.get_heart_rate()
    seq_len, num_features = train_ds[0][0].shape
    num_samples = len(train_ds)
    num_classes = 1
    return train_ds, test_ds, valid_ds, seq_len, num_features, num_samples, num_classes


def prepare_loaders(train_ds, valid_ds, test_ds, args, preprocess: bool, batch_test: int):
    if preprocess:
        full = len(train_ds)
        loaders = {
            'train': DataLoader(train_ds, batch_size=full, shuffle=True),
            'val': DataLoader(valid_ds, batch_size=len(valid_ds), shuffle=False),
            'test': DataLoader(test_ds, batch_size=len(test_ds), shuffle=False),
        }
    else:
        loaders = {
            'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            'val': DataLoader(valid_ds, batch_size=batch_test, shuffle=False),
            'test': DataLoader(test_ds, batch_size=batch_test, shuffle=False),
        }
    return loaders


def compute_and_replace_sigs(loaders, sig_fn, args):
    data = {}
    for split, loader in loaders.items():
        inputs, labels = next(iter(loader))
        start = time.time()
        inputs = sig_fn(inputs)
        print(f"{split.title()} Sigs Computed in {time.time()-start:.2f}s")
        data[split] = (inputs, labels)

    new_loaders = {}
    for split, (inp, lbl) in data.items():
        ds = TensorDataset(torch.tensor(inp).float(), torch.tensor(lbl).float())
        if split == 'train':
            new_loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        else:
            new_loaders[split] = DataLoader(ds, batch_size=100, shuffle=False)
    return new_loaders


def build_model(args, input_dim, seq_len, num_samples, num_classes, device):
    if args.model == 'transformer':
        return DecoderTransformer(
            args,
            input_dim=input_dim,
            n_head=args.n_head,
            layer=args.num_layers,
            seq_num=num_samples,
            n_embd=args.embedded_dim,
            win_len=seq_len,
            num_classes=num_classes
        ).to(device)
    elif args.model == 'lstm':
        return LSTM_Classification(
            input_size=input_dim,
            hidden_size=10,
            num_layers=100,
            batch_first=True,
            num_classes=num_classes
        ).to(device)
    raise ValueError(f"Model {args.model} not supported")


def main():
    print("Using signatures" if args.use_signatures else "Not using signatures")

    eval_bs = get_eval_batch_size(args)
    print('Dataset:', args.dataset)

    if args.dataset != 'eeg':
        raise ValueError(f"Dataset {args.dataset} not supported")

    train_ds, test_ds, valid_ds, seq_len, num_features, num_samples, num_classes = load_heart_rate_data(args)
    batch_test = 100

    loaders = prepare_loaders(train_ds, valid_ds, test_ds, args, args.preprocess, batch_test)

    if args.preprocess and args.use_signatures:
        loaders = compute_and_replace_sigs(loaders, sigs, args)

    sig_n_windows = seq_len // args.sig_win_len
    seq_len_orig = seq_len

    if args.use_signatures:
        if args.univariate:
            num_features *= args.sig_level
        else:
            num_features = signatory.signature_channels(num_features, args.sig_level)
        if args.stack:
            num_features *= 2
        seq_len = int((args.num_windows if args.irreg else seq_len // args.sig_win_len))
        print('Num features:', num_features)

    compression = ((args.num_windows if args.irreg else sig_n_windows) * num_features) / seq_len_orig
    if args.irreg:
        print(args.num_windows, num_features, seq_len_orig)

    if args.random:
        seq_len = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = build_model(args, num_features, seq_len, num_samples, num_classes, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss, val_loss = [], []
    best_val, best_epoch, no_improve = float('inf'), -1, 0
    start_time = time.time()

    for epoch in range(args.epoch):
        model.train()
        epoch_losses = []

        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            if args.random:
                indices = sorted(random.sample(range(seq_len_orig), 100))
                x = np.linspace(0, seq_len_orig, seq_len_orig)[indices]
                inputs = inputs[:, indices, :]

            if args.use_signatures and not args.preprocess:
                inputs = compute_signatures(inputs, args, x, device)

            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            epoch_losses.append(loss.item())
            train_loss.append(loss.item())

        avg_train = sum(epoch_losses) / len(epoch_losses)
        val = calculate_MSE(args, model, loaders['test'], test_ds, device)
        val_loss.append(val)

        if val < best_val:
            best_val, best_epoch, no_improve = val, epoch, 0
        else:
            no_improve += 1
            if no_improve > 15:
                args.lr /= 10
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                no_improve = 0

        if epoch - best_epoch >= args.early_stop_ep:
            print("Early stopping at epoch", epoch)
            break

        print(f"Epoch {epoch+1}/{args.epoch} | Train Loss: {avg_train:.4f} | Val Loss: {val:.4f} | Best Val: {best_val:.4f}")

    print(f"Training time: {time.time() - start_time:.2f}s")
    test_loss = calculate_MSE(args, model, loaders['test'], test_ds, device)
    print('Test Loss:', test_loss)


if __name__ == '__main__':
    main()