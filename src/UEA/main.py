import torch
print(f"Torch Version: {torch.__version__}")

import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_classification import *
from lstm_classification import *
import statistics
import numpy as np
import torch
from utils import *
import random
import argparse
import wandb
from sig_utils import ComputeSignatures
import pprint


wandb.login()

WANDB = False

# Training parameters
EPOCH = 110
BATCH_SIZE = 20
EVAL_BATCH_SIZE = -1

#Signature parameters
USE_SIGNATURES = True
SIG_WIN_LEN = 50
SIG_LEVEL = 2

NUM_WINDOWS = 100

UNIVARIATE = True

GLOBAL_BACKWARD = True
GLOBAL_FORWARD = False
LOCAL_TIGHT = False
LOCAL_WIDE = False
LOCAL_WIDTH = 50

IRREG = True
ADD_TIME = True
N_SEEDS = 5
DATASET = 'TSC_SelfRegulationSCP1'

ONLINE_SIGNATURE_CALC = False

N_HEAD = 3
NUM_LAYERS = 2
EPOCHS_FOR_CONVERGENCE = 10000
ACCURACY_FOR_CONVERGENCE = 0.6
STD_FOR_CONVERGENCE = 0.05
EMBEDDED_DIM = 10
LR = 0.00040788
WEIGHT_DECAY = 0
EMBD_PDROP = 0.1
ATTN_PDROP = 0.1
RESID_PDROP = 0.1
OVERLAP = False
SCALE_ATT = False
SPARSE = False
V_PARTITION = 0.1
Q_LEN = 1
EARLY_STOP_EP = 500
SUB_LEN = 1
WARMUP_PROPORTION = -1
OPTIMIZER = 'Adam'
CONTINUE_TRAINING = False
SAVE_ALL_EPOCHS = False
PRETRAINED_MODEL_PATH = ''
DOWNSAMPLING = False
ZERO_SHOT_DOWNSAMPLE = False
USE_RANDOM_DROP = False
RANDOM_PERCENTAGE = 0.7
MODEL = 'transformer'
TEST_SIZE = 0.3
VAL_SIZE = 0.5
INPUT_SIZE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def calculate_accuracy(config, model, data_loader, num_classes,seq_length_original, all_indices, indices_keep):
    model.eval()
    correct = 0
    total = 0
    correct = 0
    error_distribution = {}

    if config.add_time:
        t = (torch.linspace(0, seq_length_original, seq_length_original)/seq_length_original).reshape(-1,1).to(device)

    with torch.no_grad():
        for batch in data_loader:
            
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            
            if config.online_signature_calc:
                x = np.linspace(0, inputs.shape[1], inputs.shape[1])
                x = x[indices_keep]
                if config.add_time:
                    inputs = torch.cat([t.repeat(inputs.shape[0], 1, 1), inputs], dim=2)
                inputs = inputs[:,indices_keep,:]
                inputs = ComputeSignatures(inputs, x, config, device)

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


def train_transformer(config, seed):

    start_total_time = time.time()
    # print("\n".join("{}\t:\t{}".format(k, v) for k, v in config.__dict__.items()))
    print("############### Config ###############")
    if config.n_seeds == 1:
        pprint.pprint(config.__dict__['_items'])
    else:
        pprint.pprint(config.__dict__)
    print()
    print(f"Seed: {seed}") 
    print()
    print("######################################")

    if not config.online_signature_calc:
        train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features_original = get_dataset_preprocess(config,seed, device)
        num_features = num_features_original
        seq_length = seq_length_original
    else:
        train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features_original = get_dataset(config, seed)
        num_features, seq_length = ComputeModelParams(seq_length_original, num_features_original, config)

    print(f'Number of classes: {num_classes}')

    
    converged = False

    indices_keep = []
    all_indices = [i for i in range(seq_length_original)]

    if config.use_random_drop:
        print('Random')
        indices_keep = sorted(random.sample(all_indices, int(config.random_percentage*seq_length_original)))
        if not 0 in indices_keep:
            indices_keep.insert(0, 0)
    else:
        indices_keep = all_indices
    
    # Initialize the model, loss function, and optimizer
    if (config.model == 'transformer'):
            model = DecoderTransformer(
                config,
                input_dim = num_features, 
                n_head= config.n_head, 
                layer= config.num_layers, 
                seq_num = num_samples , 
                n_embd = config.embedded_dim,
                win_len= seq_length, 
                num_classes=num_classes
                ).to(device)
    elif(config.model == 'lstm'):
        model = LSTM_Classification(
            input_size=num_features, 
            hidden_size=10, 
            num_layers=100, 
            batch_first=True, 
            num_classes=num_classes
            ).to(device)
    else:
        raise ValueError('Model not supported')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    avg_epoch_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    test_accuracy_list = []
    global_step = 0
    val_accuracy_best = -float('inf')

    start_time = time.time()

    if config.add_time:
        t = (torch.linspace(0, seq_length_original, seq_length_original)/seq_length_original).reshape(-1,1).to(device)

    # Training loop
    for epoch in range(config.epoch):            
        
        epoch_loss = 0.0  

        for idx, batch in enumerate(train_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)


            if config.online_signature_calc:
                if config.add_time:
                    inputs = torch.cat([t.repeat(inputs.shape[0], 1, 1), inputs], dim=2)

                x = np.linspace(0, inputs.shape[1], inputs.shape[1])
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]
                inputs = ComputeSignatures(inputs, x, config, device)
            

            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()  # Add the loss of the current batch to the epoch loss

            
            global_step += 1
            train_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(train_loader)  # Calculate the average loss for the epoch
        if config.wandb:
            wandb.log({
                'losses/avg_epoch_loss' : avg_epoch_loss
            })

        val_accuracy, _ = calculate_accuracy(config, model, val_loader, num_classes,seq_length_original, all_indices, indices_keep)

        test_accuracy, _ = calculate_accuracy(config, model, test_loader, num_classes, seq_length_original, all_indices, indices_keep)
        
        if config.wandb:
            wandb.log({
                'accuracies/validation_accuracy': val_accuracy,
                'accuracies/test_accuracy': test_accuracy
            })     
        model.train()


        if len(val_accuracy_list) > config.epochs_for_convergence and converged == False:
            '''
            Convergence criteria: if the average validation accuracy over the last epochs_for_convergence epochs is greater than accuracy_for_convergence, 
            with a standard deviation (over the last epochs_for_convergence epochs) less than std_for_convergence * avg, we say the algorithm has converted.
            '''
            avg = statistics.mean(val_accuracy_list[-config.epochs_for_convergence:])
            std = statistics.stdev(val_accuracy_list[-config.epochs_for_convergence:])
            if avg > config.accuracy_for_convergence and std <= config.std_for_convergence * avg:
                end_time_convergence = time.time()
                convergence_time = end_time_convergence - start_time
                print(f'Model has converged at epoch {epoch}. Time taken for the model to converge: {convergence_time} seconds.')
                converged = True

        if(val_accuracy > val_accuracy_best):
            val_accuracy_best = val_accuracy
            loss_is_best = True
            best_epoch = epoch
            test_at_best_epoch = test_accuracy
        else:
            loss_is_best = False
        

        print(f'Epoch {epoch + 1}/{config.epoch}, Loss: {loss.item():.4f}, Valid Accuracy: {val_accuracy * 100:.2f}%, Best Valid Accuracy: {val_accuracy_best * 100:.2f}%, Test accuracy: {test_accuracy * 100:.2f}%' )
    
    end_time = time.time()
    execution_time = end_time - start_time
    total_time = end_time - start_total_time
    print(f"Training time: {execution_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")

    # Testing the model
    test_accuracy, test_error_distribution = calculate_accuracy(config, model, test_loader, num_classes,seq_length_original,all_indices, indices_keep)
    if config.wandb:
        wandb.log({
            'performance/best_val_accuracy': val_accuracy_best,
            'performance/test_accuracy_at_end':test_accuracy,
            'performance/best_epoch':best_epoch,
            'performance/test_at_best_epoch':test_at_best_epoch
        })
    return test_accuracy

        

def main(config=None):
    if config.wandb:
        with wandb.init(project="RFormer", config=config, allow_val_change=True):
            config = wandb.config
            if config.n_seeds == 1:
                config = wandb.config
                seed = 42
                torch.manual_seed(seed)
                print(f"Test accuracy: {train_transformer(config, seed)}")
            else:
                results = []
                for seed in range(config.n_seeds):
                    config = wandb.config
                    seed = 42+seed
                    torch.manual_seed(seed)
                    results.append(train_transformer(config, seed))
                    print(results[-1])
                results = np.array(results)
                print(f"Average accuracy: {results.mean()}, Std: {results.std()}")
                return results.mean(), results.std()
    else:
        if config.n_seeds == 1:
            seed = 42
            torch.manual_seed(seed)
            print(f"Test accuracy: {train_transformer(config, seed)}")
        else:
            results = []
            for seed in range(config.n_seeds):
                seed = 42+seed
                torch.manual_seed(seed)
                results.append(train_transformer(config, seed))
                print(results[-1])
            results = np.array(results)
            print(f"Average accuracy: {results.mean()}, Std: {results.std()}")
            return results.mean(), results.std()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--epoch", type=int, default=EPOCH, help="Number of epochs to train the model")
    parser.add_argument("--use_signatures", action='store_true' if USE_SIGNATURES else 'store_false', default=USE_SIGNATURES, help="Whether to use signatures in the model")
    parser.add_argument("--sig_win_len", type=int, default=SIG_WIN_LEN, help="Length of the signature window")
    parser.add_argument("--sig_level", type=int, default=SIG_LEVEL, help="Level of the signature")
    parser.add_argument("--num_windows", type=int, default=NUM_WINDOWS, help="Number of windows to use")
    parser.add_argument("--univariate", action='store_true' if UNIVARIATE else 'store_false', default=UNIVARIATE, help="Use univariate data")
    parser.add_argument("--irreg", action='store_true' if IRREG else 'store_false', default=IRREG, help="Use irregular intervals")
    parser.add_argument("--add_time", action='store_true' if ADD_TIME else 'store_false', default=ADD_TIME, help="Add a time channel to each input")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS, help="Number of random seeds to use")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset to use")
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE, help="Input size of the model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--n_head", type=int, default=N_HEAD, help="Number of heads in the transformer model")
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help="Number of layers in the model")
    parser.add_argument("--epochs_for_convergence", type=int, default=EPOCHS_FOR_CONVERGENCE, help="Epochs required for convergence")
    parser.add_argument("--accuracy_for_convergence", type=float, default=ACCURACY_FOR_CONVERGENCE, help="Accuracy threshold for convergence")
    parser.add_argument("--std_for_convergence", type=float, default=STD_FOR_CONVERGENCE, help="Standard deviation threshold for convergence")
    parser.add_argument("--embedded_dim", type=int, default=EMBEDDED_DIM, help="Dimension of embedded features")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay for the optimizer")
    parser.add_argument("--embd_pdrop", type=float, default=EMBD_PDROP, help="Dropout probability for embeddings")
    parser.add_argument("--attn_pdrop", type=float, default=ATTN_PDROP, help="Dropout probability for attention")
    parser.add_argument("--resid_pdrop", type=float, default=RESID_PDROP, help="Dropout probability for residual connections")
    parser.add_argument("--overlap", action='store_true' if OVERLAP else 'store_false', default=OVERLAP, help="Whether to overlap data")
    parser.add_argument("--scale_att", action='store_true' if SCALE_ATT else 'store_false', default=SCALE_ATT, help="Whether to scale attention weights")
    parser.add_argument("--sparse", action='store_true' if SPARSE else 'store_false', default=SPARSE, help="Use sparse connections")
    parser.add_argument("--v_partition", type=float, default=V_PARTITION, help="Partition ratio for validation")
    parser.add_argument("--q_len", type=int, default=Q_LEN, help="Length of the query in the model")
    parser.add_argument("--early_stop_ep", type=int, default=EARLY_STOP_EP, help="Early stopping after this many epochs")
    parser.add_argument("--sub_len", type=int, default=SUB_LEN, help="Length of the subsequences")
    parser.add_argument("--warmup_proportion", type=float, default=WARMUP_PROPORTION, help="Warmup proportion for learning rate schedule")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Optimizer to use")
    parser.add_argument("--continue_training", action='store_true' if CONTINUE_TRAINING else 'store_false', default=CONTINUE_TRAINING, help="Continue training from a checkpoint")
    parser.add_argument("--save_all_epochs", action='store_true' if SAVE_ALL_EPOCHS else 'store_false', default=SAVE_ALL_EPOCHS, help="Save model after every epoch")
    parser.add_argument("--pretrained_model_path", type=str, default=PRETRAINED_MODEL_PATH, help="Path to pretrained model")
    parser.add_argument("--downsampling", action='store_true' if DOWNSAMPLING else 'store_false', default=DOWNSAMPLING, help="Whether to downsample the data")
    parser.add_argument("--zero_shot_downsample", action='store_true' if ZERO_SHOT_DOWNSAMPLE else 'store_false', default=ZERO_SHOT_DOWNSAMPLE, help="Apply zero-shot downsampling")
    parser.add_argument("--use_random_drop", action='store_true' if USE_RANDOM_DROP else 'store_false', default=USE_RANDOM_DROP, help="Use random sampling")
    parser.add_argument("--random_percentage", type=float, default=RANDOM_PERCENTAGE, help="Percentage of data to sample randomly")
    parser.add_argument("--model", type=str, default=MODEL, help="Model type to use")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Percentage of data reserved for testing")
    parser.add_argument("--val_size", type=float, default=VAL_SIZE, help="Percentage of test set reserved for validation")
    
    parser.add_argument("--global_backward", action='store_true' if GLOBAL_BACKWARD else 'store_false', default=GLOBAL_BACKWARD, help="Use global backward sigs")
    parser.add_argument("--global_forward", action='store_true' if GLOBAL_FORWARD else 'store_false', default=GLOBAL_FORWARD, help="Use global forward sigs")
    parser.add_argument("--local_tight", action='store_true' if LOCAL_TIGHT else 'store_false', default=LOCAL_TIGHT, help="Use local tight sigs")
    parser.add_argument("--local_wide", action='store_true' if LOCAL_WIDE else 'store_false', default=LOCAL_WIDE, help="Use local wide sigs")

    parser.add_argument("--local_width", type=float, default=LOCAL_WIDTH, help="Width of local window")
    parser.add_argument("--online_signature_calc", action='store_true' if ONLINE_SIGNATURE_CALC else 'store_false', default=ONLINE_SIGNATURE_CALC, help="Use local wide sigs")
    parser.add_argument("--wandb", action='store_true' if WANDB else 'store_false', default=WANDB, help="Use wandb for logging")


    config = parser.parse_args()
    main(config)
    


