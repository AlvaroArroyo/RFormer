import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
import os
from model_classification import *
from lstm_classification import *
import argparse
import matplotlib.pyplot as plt
import statistics
import numpy as np
import iisignature
import torch
import pandas as pd
from ray import tune
from ray import train
import ray
import matplotlib.pyplot as plt
from datasets import *
from utils import *
import random

#torch.manual_seed(42)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
parser = argparse.ArgumentParser(description='PyTorch Transformer on Time series forecasting')
parser.add_argument('--input-size', default=5, type=int,
                    help='input_size (default: 5 = (4 covariates + 1 dim point))')
parser.add_argument('--batch_size', default=10, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--eval_batch_size', default=-1, type=int,
                    help='eval_batch_size default is equal to training batch_size')
parser.add_argument('--n_head', default=1, type=int,
                    help='n_head (default: 1)')
parser.add_argument('--num-layers', default=1, type=int,
                    help='num-layers (default: 1)')
parser.add_argument('--epoch', default=1000, type=int,
                    help='epoch (default: 20)')
parser.add_argument('--epochs_for_convergence', default=10000, type=int,
                    help='number of epochs evaluated to assess convergence (default: 100)')
parser.add_argument('--accuracy_for_convergence', default=0.6, type=float,
                    help='min acc in validation to assess convergence (default: 60)')
parser.add_argument('--std_for_convergence', default=0.05, type=float,
                    help='std in last epochs_for_convergence to achieve convergence')
parser.add_argument('--embedded_dim', default=20, type=int,
                    help=' The dimention of Position embedding and time series ID embedding')
parser.add_argument('--lr',default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay',default=0, type=float,
                    help='weight_decay')
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--overlap',default=False, action='store_true',
                    help='If we overlap prediction range during sampling')
parser.add_argument('--scale_att',default=False, action='store_true',
                    help='Scaling Attention')
parser.add_argument('--sparse',default=False, action='store_true',
                    help='Perform the simulation of sparse attention ')
parser.add_argument("--dataset", default='sinusoidal_long',type=str,
                    help="Dataset you want to train")
parser.add_argument('--v_partition',default=0.1, type=float,
                    help='validation_partition')
parser.add_argument('--q_len',default=1, type=int,
                    help='kernel size for generating key-query')
parser.add_argument('--early_stop_ep',default=500, type=int,
                    help='early_stop_ep')
parser.add_argument('--sub_len',default=1, type=int,
                    help='sub_len of sparse attention')
parser.add_argument('--warmup_proportion',default=-1, type=float,
                    help='warmup_proportion for BERT Adam')
parser.add_argument('--optimizer',default="Adam", type=str,
                    help='Choice BERTAdam or Adam')
parser.add_argument('--continue_training',default=False, action='store_true',
                    help='whatever to load a model and keep training it')
parser.add_argument('--save_all_epochs',default=False, action='store_true',
                    help='whatever to save the pytorch model all epochs')
parser.add_argument("--pretrained_model_path", default='',type=str,
                    help="location of the dataset to keep trainning")
parser.add_argument('--use_signatures',default=False, action='store_true',
                    help='use the signatures of the dataset or the dataset itself')
parser.add_argument("--sig_win_len", default=25,type=int,
                    help="win_len used to compute the signature")
parser.add_argument('--sig_level',default=6, type=int,
                    help='sig_level')
parser.add_argument('--hyperp_tuning',default=False, action='store_true',
                    help='whether to perform hyperparameter tuning')
parser.add_argument('--num_windows',default=75, type=int,
                    help='number of windows')
parser.add_argument('--downsampling',default=False, action='store_true',
                    help='to undersample the signal')
parser.add_argument('--zero_shot_downsample',default=False, action='store_true',
                    help='to undersample the signal at test time')
parser.add_argument('--random',default=False, action='store_true',
                    help='to drop random elements')
parser.add_argument("--model", default='transformer',type=str,
                    help="Model you want to train")
parser.add_argument('--overlapping_sigs',default=False, action='store_true',
                    help='to take overlapping signatures')
parser.add_argument('--univariate',default=False, action='store_true',
                    help='to take univariate signatures')
parser.add_argument('--stack',default=True, action='store_true',
                    help='to use multi-view attention')
parser.add_argument('--irreg',default=True, action='store_true',
                    help='to make your inputs invariant to time reparameterization')
    
def calculate_accuracy(args, model, data_loader, num_classes, indices_keep):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.zero_shot_downsample:
                if args.random:
                    indices_keep = sorted(random.sample(all_indices, 600))
                    #pass
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]
            

            if (args.use_signatures):
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

            outputs = model(inputs).to(device)
            predicted_labels = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            # Calculate error distribution
            incorrect_labels = labels[predicted_labels != labels]
            unique_labels = torch.unique(incorrect_labels)
            error_counts = torch.bincount(incorrect_labels, minlength=num_classes)
            error_distribution = {label.item(): count.item() for label, count in zip(unique_labels, error_counts) if count > 0}

    accuracy = correct / total

    return accuracy, error_distribution


# Compute (S(X)_0,t_1, S(X)_0,t_2, ...)
# Data - (B, T, F)
# Fixed sig window length
def Signature_overlapping_univariate(data, depth, sig_window, device):

    B, T, F = data.shape

    # (B, T, F) -> (B, T-1, F_sig)
    # sigs = [iisignature.sig(data.cpu(), depth, 2)]

    sigs = [iisignature.sig(data.cpu()[:, :, i].unsqueeze(2), depth, 2) for i in range(F)]
    sigs = np.concatenate(sigs, 2)
    sigs = torch.tensor(sigs).to(device)

    # Select indices of desired signatures
    indices = np.arange(sig_window-2, data.shape[1], sig_window)
    
    # (B, T-1, F_sig) -> (B, T_sig, F_sig)
    return sigs[:, indices, :].to(torch.float32)


# Compute (S(X)_0,t_1, S(X)_t_1,t_2, ...)
# Data - (B, T, F)
# Fixed sig window length
#THIS SHOULD BE REVISED TO AVOID THE PYTORCH -> NUMPY -> PYTORCH TRANSITION
def Signature_nonoverlapping_univariate(data, depth, sig_win_length, device):
    B, T, F = data.shape[0], data.shape[1], data.shape[2]
    n_windows = int(T/sig_win_length)

    indices = np.arange(sig_win_length-2, data.shape[1], sig_win_length)
    data_ = data[:, :(indices[-1]+2), :]
    data_ = data_.reshape(B, n_windows, -1, F).cpu()

    # (B, T, F) -> (B, T_sig, F_sig)
    sigs = [iisignature.sig(data_[:, :, :, _].unsqueeze(3), depth) for _ in range(F)]
    sigs = np.concatenate(sigs, 2)
    return torch.Tensor(sigs).to(device).to(torch.float32)

# Compute (S(X)_0,t_1, S(X)_0,t_2, ...)
# Data - (B, T, F)
# Fixed sig window length
def Signature_overlapping(data, depth, sig_window, device):

    # (B, T, F) -> (B, T-1, F_sig)
    sigs = iisignature.sig(data.cpu(), depth, 2)

    # Select indices of desired signatures
    indices = np.arange(sig_window-2, data.shape[1], sig_window)
    
    # (B, T-1, F_sig) -> (B, T_sig, F_sig)
    return torch.Tensor(sigs[:, indices, :]).to(device)


def Signature_overlapping_irreg(data, depth, num_windows, x, device):
    '''
        data: signal over whic to take the signature
        depth: signature depth
        num_windows: number of windows (equally spaced)
        x: 
    '''
    step = max(x)/num_windows


    # (B, T, F) -> (B, T-1, F_sig)
    # This function takes the signature at every point
    sigs = iisignature.sig(data.cpu(), depth, 2)

    # We now pick the signatures according to the indices to be robust to the sampling
    indices = [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)]
    indices[-1] -= 1
    
    # (B, T-1, F_sig) -> (B, T_sig, F_sig)
    return torch.Tensor(sigs[:, indices, :]).to(device)


# Compute (S(X)_0,t_1, S(X)_t_1,t_2, ...)
# Data - (B, T, F)
# Fixed sig window length
#THIS SHOULD BE REVISED TO AVOID THE PYTORCH -> NUMPY -> PYTORCH TRANSITION
def Signature_nonoverlapping(data, depth, sig_win_length, device):
    B, T, F = data.shape[0], data.shape[1], data.shape[2]
    n_windows = int(T/sig_win_length)

    # (B, T, F) -> (B, T_sig, F_sig)
    sigs = iisignature.sig(data.reshape(B, n_windows, -1, F).cpu(), depth)
    return torch.Tensor(sigs).to(device)


def Signature_nonoverlapping_irreg(data, depth, num_windows, x, device):
    '''
        data: signal over whic to take the signature
        depth: signature depth
        num_windows: number of windows (equally spaced)
        x: 
    '''
    step = max(x)/num_windows
    indices = [0] + [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)] 

    data = data.cpu()

    for i in range(len(indices)-1):
        slice = data[:,indices[i]:indices[i+1],:]
        sig_slice = iisignature.sig(slice, depth).reshape(data.shape[0], 1, -1)
        if i == 0:
            sigs = sig_slice
        else:
            sigs = np.concatenate((sigs, sig_slice), axis=1)
    return torch.Tensor(sigs).to(device)


global args
args = parser.parse_args()

def main(hyperp_tuning=False):

    print('Using signatures') if args.use_signatures else print('Not using signatures')
    print(args)

    # Create dataset and data loader
    if(args.eval_batch_size ==-1):
        eval_batch_size = args.batch_size
    else:
        eval_batch_size = args.eval_batch_size

    print('Dataset', args.dataset)
    
    if (args.dataset == 'sinusoidal_long'):
        # Hyperparameters for the dataset and dataloader
        num_samples = 1000
        seq_length = 2000
        seq_length_orig = seq_length

    
        num_features = 2
        '''
        freq_min=1 
        freq_max= 11
        num_classes=10
        '''

        freq_min=10 
        freq_max=500 
        num_classes=100
    
        dataset = SinusoidalDatasetLong(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = SinusoidalDatasetLong(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        # Create test dataset and loader
        test_dataset = SinusoidalDatasetLong(num_samples, seq_length, num_features, freq_min, freq_max, num_classes)  # 100 test samples
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    else:
        raise ValueError('Dataset not supported')
    

    print('Num features: ', num_features)

    if (args.use_signatures):
        if not args.univariate:
            sig_output_size = signature_channels(num_features, args.sig_level)
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
    else:
        if args.random:
            seq_length = 600
    
    
    sig_n_windows = int(seq_length/args.sig_win_len)

    #sig_features = num_features * args.sig_level #new


    print('Num features: ', num_features)
    print('Stack: ', args.stack)

    if not args.irreg:
        compression = (sig_n_windows*num_features)/seq_length_orig
    else:
        compression = (args.num_windows*num_features)/seq_length_orig
        print(args.num_windows, num_features, seq_length_orig)
    converged = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    if not hyperp_tuning:
        date_log = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        if (args.use_signatures):
            if not args.irreg:
                logs_name = f'{args.dataset}_{date_log}_sig_win_len={args.sig_win_len}_[#W={sig_n_windows}]_sig_level={args.sig_level}_compression={compression}_overlapping={args.overlapping_sigs}_stack={args.stack}'
            else:
                if args.random:
                    logs_name = f'{args.dataset}_{date_log}_num_win={args.num_windows}_sig_level={args.sig_level}_compression={compression}_overlapping={args.overlapping_sigs}_stack={args.stack}_random'
                elif args.downsampling:
                    logs_name = f'{args.dataset}_{date_log}_num_win={args.num_windows}_sig_level={args.sig_level}_compression={compression}_overlapping={args.overlapping_sigs}_stack={args.stack}_zs'
                else:
                    logs_name = f'{args.dataset}_{date_log}_num_win={args.num_windows}_sig_level={args.sig_level}_compression={compression}_overlapping={args.overlapping_sigs}_stack={args.stack}'
        else:
            if args.random:
                logs_name = f'{args.dataset}_{date_log}_use_signatures={args.use_signatures}_model={args.model}_random'
            elif args.downsampling:
                logs_name = f'{args.dataset}_{date_log}_use_signatures={args.use_signatures}_model={args.model}_downsample'
            else:
                logs_name = f'{args.dataset}_{date_log}_use_signatures={args.use_signatures}_model={args.model}'
  
        print (logs_name)
        outdir = f'exp_{logs_name}'
        os.mkdir(outdir)
        writer = SummaryWriter(outdir) 

    


    global indices_keep
    indices_keep = []
    global all_indices
    all_indices = [i for i in range(seq_length_orig)]
    if args.downsampling:
        print('Downsampling')
        for idx in range(seq_length_orig):
            if idx % 2 == 0:
                indices_keep.append(idx)
    elif args.random:
        print('Random')
        for idx in range(seq_length_orig):
            indices_keep = sorted(random.sample(all_indices, 1000))
    
    # Initialize the model, loss function, and optimizer
    if (args.model == 'transformer'):
            model = DecoderTransformer(args,input_dim = num_features, n_head= args.n_head, layer= args.num_layers, seq_num = num_samples , n_embd = args.embedded_dim,win_len= seq_length, num_classes=num_classes).to(device)
    elif(args.model == 'lstm'):
        model = LSTM_Classification(input_size=num_features, hidden_size=10, num_layers=100, batch_first=True, num_classes=num_classes).to(device)
    else:
        raise ValueError('Model not supported')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss_list = []
    val_accuracy_list = []
    test_accuracy_list = []
    global_step = 0
    val_accuracy_best = -float('inf')

    start_time = time.time()

 
    # Training loop
    for epoch in range(args.epoch):            
        epoch_loss = 0.0  # Variable to store the total loss for the epoch
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.random:
                indices_keep = sorted(random.sample(all_indices, 600))
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]
        
            if (args.use_signatures):
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
      

            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()  # Add the loss of the current batch to the epoch loss
            writer.add_scalar('training/train_loss', loss, global_step) if not hyperp_tuning else train.report({"training/train_loss": loss.item()})

            
            global_step += 1
            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(data_loader)  # Calculate the average loss for the epoch
        writer.add_scalar('training/avg_train_loss', avg_epoch_loss, epoch) if not hyperp_tuning else train.report({"training/avg_train_loss": avg_epoch_loss}) # Add the average loss to the writer
        val_accuracy, _ = calculate_accuracy(args, model, val_loader, num_classes, indices_keep)
        test_accuracy, _ = calculate_accuracy(args, model, test_loader, num_classes, indices_keep)
        model.train()
        train.report({"val_accuracy": val_accuracy})
        writer.add_scalar('training/val_accuracy', val_accuracy, epoch) if not hyperp_tuning else train.report({"training/val_accuracy": val_accuracy})
        val_accuracy_list.append(val_accuracy)

        length = 250
        if args.use_signatures:
            pd.Series(val_accuracy_list).to_csv(f'val_acc_sinelong_s_{args.sig_level}_w{args.num_windows}.csv')
        else:
            pd.Series(val_accuracy_list).to_csv(f'val_acc_long_transformer_time_{length}.csv')



        train.report({"test_accuracy": test_accuracy})
        writer.add_scalar('training/test_accuracy', test_accuracy, epoch) if not hyperp_tuning else train.report({"training/test_accuracy": val_accuracy})
        test_accuracy_list.append(test_accuracy)

        if len(val_accuracy_list) > args.epochs_for_convergence and converged == False:
            '''
            Convergence criteria: if the average validation accuracy over the last epochs_for_convergence epochs is greater than accuracy_for_convergence, 
            with a standard deviation (over the last epochs_for_convergence epochs) less than std_for_convergence * avg, we say the algorithm has converted.
            '''
            avg = statistics.mean(val_accuracy_list[-args.epochs_for_convergence:])
            std = statistics.stdev(val_accuracy_list[-args.epochs_for_convergence:])
            if avg > args.accuracy_for_convergence and std <= args.std_for_convergence * avg:
                end_time_convergence = time.time()
                convergence_time = end_time_convergence - start_time
                writer.add_scalar('training/time_convergence', convergence_time, epoch) if not hyperp_tuning else train.report({"training/time_convergence": convergence_time})
                print(f'Model has converged at epoch {epoch}. Time taken for the model to converge: {convergence_time} seconds.')
                converged = True

        if(val_accuracy > val_accuracy_best):
            val_accuracy_best = val_accuracy
            loss_is_best = True
            best_epoch = epoch
        else:
            loss_is_best = False
        
        #if(epoch-best_epoch>=args.early_stop_ep):
        #    print("Achieve early_stop_ep and current epoch is",epoch)
        #    break
        if not hyperp_tuning:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer' : model.state_dict(),
            },epoch+1,loss_is_best,outdir,args.save_all_epochs)

        print(f'Epoch {epoch + 1}/{args.epoch}, Loss: {loss.item():.4f}, Valid Accuracy: {val_accuracy * 100:.2f}%, Best Valid Accuracy: {val_accuracy_best * 100:.2f}%')

    end_time = time.time()
    execution_time = end_time - start_time
    writer.add_scalar('training/time_training', execution_time, epoch) if not hyperp_tuning else train.report({"training/time_training":execution_time })
    print(f"Training time: {execution_time:.2f} seconds")

    if (not hyperp_tuning):
        model.load_state_dict(torch.load(outdir+'/best_v_loss.pth.tar')['state_dict']) if not hyperp_tuning else None
        # Testing the model
        test_accuracy, test_error_distribution = calculate_accuracy(args, model, test_loader, num_classes, indices_keep)
        if (args.use_signatures):
            print(f'Test Accuracy using signatures: {test_accuracy * 100:.2f}%')
        else:
            print(f'Test Accuracy not using signatures: {test_accuracy * 100:.2f}%')

        writer.add_scalar('test/val_accuracy', test_accuracy, epoch)


            
        #plot_error_distribution(test_error_distribution, outdir, args.use_signatures)
        plot_functions(train_loss_list, val_accuracy_list, outdir, args.use_signatures, test_accuracy_list)
        plot_predictions_signatures(device, args, model, test_loader, outdir, num_predictions=2, all_indices=all_indices) if args.use_signatures else plot_predictions(device, model, test_loader, outdir, num_predictions=10, all_indices=all_indices)

if __name__ == '__main__':
    #main(hyperp_tuning=args.hyperp_tuning)

    if (args.hyperp_tuning):
        ray.init(ignore_reinit_error=True)
        local_dir="/nfs/home/fernandom/github/signatures/hyperp_tuning"
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
        analysis = tune.run(
            main,
            config={
                "sig_level": tune.grid_search([2]),
                "sig_win_len": tune.grid_search([2]),
                "num_epochs": args.epoch  # Specify the number of epochs here
            },
            resources_per_trial={"cpu": 1, "gpu": 1},
            keep_checkpoints_num=1,
            checkpoint_score_attr="val_accuracy",
            local_dir=local_dir,
        )
        print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))
        '''
        We reload the best sig_level and sig_win_len to perform a final training with the best hyperparameters.
        PS: We could also use the best checkpoint, but we are training from scratch to get all metrics regarding convergence and training time.
        '''
        args.sig_level = analysis.get_best_config(metric="val_accuracy", mode="max")["sig_level"]
        args.sig_win_len = analysis.get_best_config(metric="val_accuracy", mode="max")["sig_win_len"]
        args.hyperp_tuning = False
        main(hyperp_tuning=args.hyperp_tuning)
    else:
        main(hyperp_tuning=args.hyperp_tuning)