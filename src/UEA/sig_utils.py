import iisignature
import numpy as np
import torch



def global_signature_backward(data, x, num_windows, depth, univariate, device):
    F = data.shape[-1]
    step = max(x)/num_windows

    indices = [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)]
    indices[-1] -= 1

    if univariate:
        sigs = [torch.tensor(iisignature.sig(torch.cat([data[:, :, 0:1], data[:, :, i:i+1]], dim=2).cpu(), depth, 2)) for i in range(1,F)]
        sigs = torch.cat(sigs, dim=2).to(device)
    else:
        # (B, T, F) -> (B, T-1, F_sig)
        sigs = torch.tensor(iisignature.sig(data.cpu(), depth, 2))

    # (B, T-1, F_sig) -> (B, T_sig, F_sig)
    output = sigs[:, indices, :].to(device)
    return output

def global_signature_forward(data, x, num_windows, depth, univariate, device):
    step = max(x)/num_windows
    indices = [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)]
    indices[-1] -= 1
    F = data.shape[-1]
    flipped_data = torch.flip(data, dims=[1]).cpu()

    if univariate:
        sigs = [torch.tensor(iisignature.sig(torch.cat([flipped_data[:, :, 0:1], flipped_data[:, :, i:i+1]], dim=2).cpu(), depth, 2)) for i in range(1,F)]
        sigs = torch.cat(sigs, dim=2).to(device)
    else:
        # (B, T, F) -> (B, T-1, F_sig)
        sigs = torch.tensor(iisignature.sig(flipped_data.cpu(), depth, 2))
    
    # (B, T-1, F_sig) -> (B, T_sig, F_sig)
    return torch.Tensor(sigs[:, indices, :]).to(device)

def local_signature_tight(data, x, num_windows, depth, univariate, device):
    '''
        data: signal over whic to take the signature
        depth: signature depth
        num_windows: number of windows (equally spaced)
        x: 
    '''
    step = max(x)/num_windows
    indices = [0] + [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)] 
    indices[-1] -= 1
    F = data.shape[-1]

    data = data.cpu()

    if univariate:
        for i in range(len(indices)-1):
            slice = data[:,indices[i]:indices[i+1],:]
            sig_slice = np.concatenate([np.expand_dims(iisignature.sig(torch.cat([slice[:, :, 0:1], slice[:, :, _:_+1]], dim=2).cpu(), depth), axis=1) for _ in range(1, F)], axis=2)
            if i == 0:
                sigs = sig_slice
            else:
                sigs = np.concatenate((sigs, sig_slice), axis=1)
    else:
        for i in range(len(indices)-1):
            slice = data[:,indices[i]:indices[i+1],:]
            sig_slice = np.expand_dims(iisignature.sig(slice, depth), axis=1)
            if i == 0:
                sigs = sig_slice
            else:
                sigs = np.concatenate((sigs, sig_slice), axis=1)
    return torch.Tensor(sigs).to(device)

def local_signature_wide(data, x, num_windows, width, depth, univariate, device):
    step = max(x)/num_windows
    indices = [0] + [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)] 
    indices[-1] -= 1
    F = data.shape[-1]

    data = data.cpu()
    if univariate:
        for i in range(len(indices)-1):
            start = max(0, indices[i] - width)
            end = min(int(max(x)), indices[i+1]+width)
            slice = data[:,start:end,:]
            sig_slice = np.concatenate([np.expand_dims(iisignature.sig(torch.cat([slice[:, :, 0:1], slice[:, :, _:_+1]], dim=2).cpu(), depth), axis=1) for _ in range(1, F)], axis=2)
            if i == 0:
                sigs = sig_slice
            else:
                sigs = np.concatenate((sigs, sig_slice), axis=1)
    else:
        for i in range(len(indices)-1):
            start = max(0, indices[i] - width)
            end = min(int(max(x)), indices[i+1]+width)
            slice = data[:,start:end,:]
            sig_slice = np.expand_dims(iisignature.sig(slice, depth), axis=1)
            if i == 0:
                sigs = sig_slice
            else:
                sigs = np.concatenate((sigs, sig_slice), axis=1)
    return torch.Tensor(sigs).to(device)

def ComputeSignatures(inputs, x, config, device):
    output = []
    if config.global_backward:
        output.append(global_signature_backward(inputs, x, config.num_windows, config.sig_level, config.univariate, device))
    if config.global_forward:
        output.append(global_signature_forward(inputs, x, config.num_windows, config.sig_level, config.univariate, device))
    if config.local_tight:
        output.append(local_signature_tight(inputs, x, config.num_windows, config.sig_level, config.univariate, device))
    if config.local_wide:
        output.append(local_signature_wide(inputs, x, config.num_windows, config.local_width, config.sig_level, config.univariate, device))
    return torch.cat(output, dim=2).float()





# # Compute (S(X)_0,t_1, S(X)_0,t_2, ...)
# # Data - (B, T, F)
# # Fixed sig window length
# def Signature_overlapping_univariate(data, depth, sig_window, device):

#     B, T, F = data.shape

#     # (B, T, F) -> (B, T-1, F_sig)
#     # sigs = [iisignature.sig(data.cpu(), depth, 2)]

#     sigs = [iisignature.sig(data.cpu()[:, :, i].unsqueeze(2), depth, 2) for i in range(F)]
#     sigs = np.concatenate(sigs, 2)
#     sigs = torch.tensor(sigs).to(device)

#     # Select indices of desired signatures
#     indices = np.arange(sig_window-2, data.shape[1], sig_window)
    
#     # (B, T-1, F_sig) -> (B, T_sig, F_sig)
#     return sigs[:, indices, :].to(torch.float32)


# def Signature_overlapping_univariate_irreg(data, depth, num_windows, x, device):
#     '''
#         data: signal over whic to take the signature
#         depth: signature depth
#         num_windows: number of windows (equally spaced)
#         x: 
#     '''
#     B, T, F = data.shape[0], data.shape[1], data.shape[2]

#     step = max(x)/num_windows

#     # (B, T, F) -> (B, T-1, F_sig)
#     # This function takes the signature at every point
#     sigs = [iisignature.sig(data.cpu()[:, :, i].unsqueeze(2), depth, 2) for i in range(F)]
#     sigs = np.concatenate(sigs, 2)
#     sigs = torch.tensor(sigs).to(device)

#     # We now pick the signatures according to the indices to be robust to the sampling
#     indices = [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)]
#     indices[-1] -= 1
    
#     # (B, T-1, F_sig) -> (B, T_sig, F_sig)
#     return sigs[:, indices, :].to(torch.float32)

# # Compute (S(X)_0,t_1, S(X)_t_1,t_2, ...)
# # Data - (B, T, F)
# # Fixed sig window length
# def Signature_nonoverlapping_univariate(data, depth, sig_win_length, device):
#     B, T, F = data.shape[0], data.shape[1], data.shape[2]
#     n_windows = int(T/sig_win_length)

#     indices = np.arange(sig_win_length-2, data.shape[1], sig_win_length)
#     data_ = data[:, :(indices[-1]+2), :]
#     data_ = data_.reshape(B, n_windows, -1, F).cpu()

#     # (B, T, F) -> (B, T_sig, F_sig)
#     sigs = [iisignature.sig(data_[:, :, :, _].unsqueeze(3), depth) for _ in range(F)]
#     sigs = np.concatenate(sigs, 2)
#     return torch.Tensor(sigs).to(device).to(torch.float32)

# # Compute (S(X)_0,t_1, S(X)_0,t_2, ...)
# # Data - (B, T, F)
# # Fixed sig window length
# def Signature_overlapping(data, depth, sig_window, device):

#     # (B, T, F) -> (B, T-1, F_sig)
#     sigs = iisignature.sig(data.cpu(), depth, 2)

#     # Select indices of desired signatures
#     indices = np.arange(sig_window-2, data.shape[1], sig_window)
    
#     # (B, T-1, F_sig) -> (B, T_sig, F_sig)
#     return torch.Tensor(sigs[:, indices, :]).to(device)


# def Signature_overlapping_irreg(data, depth, num_windows, x, device):
#     '''
#         data: signal over whic to take the signature
#         depth: signature depth
#         num_windows: number of windows (equally spaced)
#         x: 
#     '''
#     step = max(x)/num_windows


#     # (B, T, F) -> (B, T-1, F_sig)
#     # This function takes the signature at every point
#     sigs = iisignature.sig(data.cpu(), depth, 2)

#     # We now pick the signatures according to the indices to be robust to the sampling
#     indices = [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)]
#     indices[-1] -= 1
    
#     # (B, T-1, F_sig) -> (B, T_sig, F_sig)
#     return torch.Tensor(sigs[:, indices, :]).to(device)

# # Compute (S(X)_0,t_1, S(X)_t_1,t_2, ...)
# # Data - (B, T, F)
# def Signature_nonoverlapping(data, depth, sig_win_length, device):
#     B, T, F = data.shape[0], data.shape[1], data.shape[2]
#     n_windows = int(T/sig_win_length)

#     # (B, T, F) -> (B, T_sig, F_sig)
#     sigs = iisignature.sig(data.reshape(B, n_windows, -1, F).cpu(), depth)
#     return torch.Tensor(sigs).to(device)


# def Signature_nonoverlapping_irreg(data, depth, num_windows, x, device):
#     '''
#         data: signal over whic to take the signature
#         depth: signature depth
#         num_windows: number of windows (equally spaced)
#         x: 
#     '''
#     step = max(x)/num_windows
#     indices = [0] + [np.where(x < step*i)[0][-1] for i in range(1, num_windows+1)] 

#     data = data.cpu()

#     for i in range(len(indices)-1):
#         slice = data[:,indices[i]:indices[i+1],:]
#         sig_slice = iisignature.sig(slice, depth).reshape(data.shape[0], 1, -1)
#         if i == 0:
#             sigs = sig_slice
#         else:
#             sigs = np.concatenate((sigs, sig_slice), axis=1)
#     return torch.Tensor(sigs).to(device)

# def ComputeSignatures(inputs, x, config, device):
#     if not config.stack:
#         if config.overlapping_sigs and config.univariate and not config.irreg:
#             inputs=Signature_overlapping_univariate(inputs,config.sig_level, config.sig_win_len, device)

#         elif config.overlapping_sigs and config.univariate and config.irreg:
#             inputs=Signature_overlapping_univariate_irreg(inputs,config.sig_level, config.num_windows, x, device)

#         elif config.overlapping_sigs and not config.univariate:
#             if config.irreg:
#                 inputs = Signature_overlapping_irreg(inputs,config.sig_level, config.num_windows, x, device)
#             else:
#                 inputs = Signature_overlapping(inputs, config.sig_level, config.sig_win_len, device)
#         elif not config.overlapping_sigs and config.univariate:
#             if config.irreg:  ## fix this
#                 inputs = Signature_nonoverlapping_univariate(inputs,config.sig_level, config.sig_win_len, device)
#             else:
#                 inputs = Signature_nonoverlapping_univariate(inputs,config.sig_level, config.sig_win_len, device)
#         elif not config.overlapping_sigs and not config.univariate:
#             if config.irreg:
#                 inputs = Signature_nonoverlapping_irreg(inputs,config.sig_level, config.num_windows, x, device)
#             else:
#                 inputs = Signature_nonoverlapping(inputs,config.sig_level, config.sig_win_len, device)
#     else:
#         if not config.univariate:
#             if config.irreg:
#                 inputs1 = Signature_nonoverlapping_irreg(inputs,config.sig_level, config.num_windows, x, device)
#                 inputs2 = Signature_overlapping_irreg(inputs,config.sig_level, config.num_windows, x, device)
#             else:
#                 inputs1 = Signature_nonoverlapping(inputs,config.sig_level, config.sig_win_len, device)
#                 inputs2 = Signature_overlapping(inputs,config.sig_level, config.sig_win_len, device)
#             inputs = torch.cat((inputs1, inputs2), dim=2)
            
#         else:
#             inputs1 = Signature_nonoverlapping_univariate(inputs,config.sig_level, config.sig_win_len, device)
#             inputs2 = Signature_overlapping_univariate(inputs,config.sig_level, config.sig_win_len, device)
#             inputs = torch.cat((inputs1, inputs2), dim=2)
