from sig_utils import *
import torch.nn as nn

def calculate_MSE(args, model, data_loader, dataset, device):
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