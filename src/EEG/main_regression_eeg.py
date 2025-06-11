import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from model_classification import *
import argparse
import matplotlib.pyplot as plt
import statistics
import numpy as np
import iisignature
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import *
import signatory
from sig_utils import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
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
    
def calculate_MSE(args, model, data_loader, dataset):
    model.eval()
    loss = 0.0
    objective_test = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for batch in data_loader:
        
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.zero_shot_downsample:
                if args.random:
                    indices_keep = sorted(random.sample(all_indices, 100))
                    #pass
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]

            if (args.use_signatures) and not args.preprocess:
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


            outputs = model(inputs).squeeze(-1).to(device)
            loss += objective_test(outputs, labels)

        loss /= len(dataset)
        loss = torch.sqrt(loss)
            
    return loss.item()


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


global args
args = parser.parse_args()
print(args)

def main():

    print('Using signatures') if args.use_signatures else print('Not using signatures')

    # Create dataset and data loader
    if(args.eval_batch_size ==-1):
        eval_batch_size = args.batch_size
    else:
        eval_batch_size = args.eval_batch_size

    print('Dataset', args.dataset)
    
    if args.dataset == 'eeg':
        eigenworms = HeartRate()
        train_dataset, test_dataset, valid_dataset = eigenworms.get_heart_rate()

        seq_length = train_dataset[0][0].shape[0]
        seq_length_orig = seq_length
        num_features = train_dataset[0][0].shape[1]
        num_samples = len(train_dataset)
        num_classes = 1
        batch_test = 100

        if args.preprocess:
            data_loader = DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset))
            val_loader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
        else:
            data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
            val_loader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))

        if args.preprocess:
            for idx, batch in enumerate(data_loader):
                inputs_train, labels_train = batch
            for idx, batch in enumerate(val_loader):
                inputs_val, labels_val = batch
            for idx, batch in enumerate(test_loader):
                inputs_test, labels_test = batch
            
            init_time_train = time.time()
            inputs_train = sigs(inputs_train)
            print('Train Sigs Computed, Time taken: ', time.time() - init_time_train)
            init_time_val = time.time()
            inputs_val = sigs(inputs_val)
            print('Val Sigs Computed, Time taken: ', time.time() - init_time_val)
            init_time_test = time.time()
            inputs_test = sigs(inputs_test)
            print('Test Sigs Computed, Time taken: ', time.time() - init_time_test)


            train_dataset = TensorDataset(Tensor(inputs_train).float(), Tensor(labels_train).float())
            valid_dataset = TensorDataset(Tensor(inputs_val).float(), Tensor(labels_val).float())
            test_dataset = TensorDataset(Tensor(inputs_test).float(), Tensor(labels_test).float())

            data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
            val_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_test)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)           

    else:
        raise ValueError('Dataset not supported')
    

    sig_n_windows = int(seq_length/args.sig_win_len)
    #sig_features = signatory.signature_channels(num_features, args.sig_level)
    #sig_features = num_features * args.sig_level #new

    if (args.use_signatures):
        if not args.univariate:
            sig_output_size = signatory.signature_channels(num_features, args.sig_level)
            num_features = sig_output_size
        else:
            num_features = num_features * args.sig_level

        if args.stack:
            num_features *= 2

        print('Num features: ', num_features)
        
        if args.irreg:
            seq_length = int(args.num_windows) 
        else:
            seq_length = int(seq_length/args.sig_win_len)

    if not args.irreg:
        compression = (sig_n_windows*num_features)/seq_length_orig
    else:
        compression = (args.num_windows*num_features)/seq_length_orig
        print(args.num_windows, num_features, seq_length_orig)
    
    if args.random:
        seq_length = 100

    converged = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    

    

   


    # Initialize the model, loss function, and optimizer
    if (args.model == 'transformer'):
        model = DecoderTransformer(args,input_dim = num_features, n_head= args.n_head, layer= args.num_layers, seq_num = num_samples , n_embd = args.embedded_dim,win_len= seq_length, num_classes=num_classes).to(device)
    elif(args.model == 'lstm'):
        model = LSTM_Classification(input_size=num_features, hidden_size=10, num_layers=100, batch_first=True, num_classes=num_classes).to(device)
    else:
        raise ValueError('Model not supported')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_list = []
    val_loss_list = []
    global_step = 0
    val_loss_best = float('inf')

    start_time = time.time()
    epochs_since_last_improvement = 0

    global indices_keep
    indices_keep = []
    global all_indices
    all_indices = [i for i in range(seq_length_orig)]
    for idx in range(train_dataset[0][0].shape[0]):
        if idx % 2 == 0:
            indices_keep.append(idx)

    # Training loop
    for epoch in range(args.epoch):            
        epoch_loss = 0.0  # Variable to store the total loss for the epoch
        start_time_epoch = time.time()
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.random:
                indices_keep = sorted(random.sample(all_indices, 100))
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]

            model.train() 
            
            if (args.use_signatures) and not args.preprocess:
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


            outputs = model(inputs).squeeze(-1).to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()  # Add the loss of the current batch to the epoch loss

            
            global_step += 1
            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time_epoch = time.time()
        avg_epoch_loss = epoch_loss / len(data_loader)  # Calculate the average loss for the epoch
   
        val_loss = calculate_MSE(args, model, test_loader, test_dataset)
     
        val_loss_list.append(val_loss)

        if(val_loss < val_loss_best):
            val_loss_best = val_loss
            loss_is_best = True
            best_epoch = epoch
            epochs_since_last_improvement = 0
        else:
            loss_is_best = False
            epochs_since_last_improvement += 1
        
        if epochs_since_last_improvement > 15:
            epochs_since_last_improvement = 0
            print("Reducing Learning Rate")
            args.lr /= 10.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    
        if(epoch-best_epoch>=args.early_stop_ep):
            print("Achieve early_stop_ep and current epoch is",epoch)
            break
       
    
        print(f'Epoch {epoch + 1}/{args.epoch}, Loss: {loss.item():.4f}, Valid Loss: {val_loss:.2f}, Best Valid Loss: {val_loss_best:.2f}, Epoch Time: {(end_time_epoch - start_time_epoch):.2f}')

  

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training time: {execution_time:.2f} seconds")


    test_loss = calculate_MSE(args, model, test_loader, test_dataset)
    print('Test Loss: ', test_loss)
    train.report({"test_loss": test_loss})
    plot_predictions_signatures(device, args, model, test_loader, outdir, num_predictions=2) if args.use_signatures else plot_predictions(device, model, test_loader, outdir, num_predictions=10, all_indices=all_indices)

if __name__ == '__main__':
    main()