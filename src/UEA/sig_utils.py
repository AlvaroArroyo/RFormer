import iisignature
import numpy as np
import torch
from typing import List

# Configuration object placeholder for type hinting
class SignatureConfig:
    global_backward: bool
    global_forward: bool
    local_tight: bool
    local_wide: bool
    num_windows: int
    sig_level: int
    univariate: bool
    local_width: int

def _get_signature(
    path: torch.Tensor, 
    depth: int, 
    univariate: bool, 
    streamed: bool = False
) -> np.ndarray:
    """
    A helper function that computes a signature for a given path.

    It handles CPU conversion, univariate vs. multivariate cases, and streamed
    vs. non-streamed signature calculations.
    """
    path_cpu = path.cpu()
    sig_opts = (depth, 2) if streamed else (depth,)

    if univariate:
        # For univariate mode, compute signatures for 2D paths composed of the
        # base channel (assumed to be time) and each feature channel.
        base_channel = path_cpu[..., 0:1]
        num_features = path_cpu.shape[-1]
        
        # iisignature returns numpy arrays, so we concatenate with numpy
        sigs = [
            iisignature.sig(torch.cat([base_channel, path_cpu[..., i:i+1]], dim=-1), *sig_opts)
            for i in range(1, num_features)
        ]
        return np.concatenate(sigs, axis=-1)
    else:
        # For the multivariate case, compute the signature on the full path.
        return iisignature.sig(path_cpu, *sig_opts)

# --- Global Signature Functions ---

def _compute_global_sigs(
    data: torch.Tensor, 
    x: np.ndarray, 
    num_windows: int, 
    depth: int, 
    univariate: bool, 
    device: torch.device, 
    reverse_stream: bool
) -> torch.Tensor:
    """Internal helper to compute forward and backward global signatures."""
    # The 'forward' signature processes the time-reversed path
    path = torch.flip(data, dims=[1]) if reverse_stream else data
    
    step = max(x) / num_windows
    indices = [np.where(x < step * i)[0][-1] for i in range(1, num_windows + 1)]
    indices[-1] -= 1  # Preserving original index adjustment

    # Get the streamed signature of the entire path
    sigs_stream = _get_signature(path, depth, univariate, streamed=True)
    
    # Select signature states at window endpoints and move to the target device
    output = torch.from_numpy(sigs_stream)[:, indices, :]
    return output.to(device)

def global_signature_backward(
    data: torch.Tensor, x: np.ndarray, num_windows: int, 
    depth: int, univariate: bool, device: torch.device
) -> torch.Tensor:
    """Computes the global signature on the data stream."""
    return _compute_global_sigs(data, x, num_windows, depth, univariate, device, reverse_stream=False)

def global_signature_forward(
    data: torch.Tensor, x: np.ndarray, num_windows: int, 
    depth: int, univariate: bool, device: torch.device
) -> torch.Tensor:
    """Computes the global signature on the time-reversed data stream."""
    return _compute_global_sigs(data, x, num_windows, depth, univariate, device, reverse_stream=True)

# --- Local Signature Functions ---

def _compute_local_sigs(
    data: torch.Tensor, 
    x: np.ndarray, 
    num_windows: int, 
    depth: int, 
    univariate: bool, 
    device: torch.device, 
    width: int
) -> torch.Tensor:
    """Internal helper for both tight and wide local signatures."""
    step = max(x) / num_windows
    # Define window boundaries, starting from index 0
    indices = [0] + [np.where(x < step * i)[0][-1] for i in range(1, num_windows + 1)]
    indices[-1] -= 1  # Preserving original index adjustment

    all_sigs = []
    # Loop over windows and compute the signature for each corresponding slice
    for i in range(len(indices) - 1):
        start = max(0, indices[i] - width)
        # Preserving original clipping logic
        end = min(int(max(x)), indices[i+1] + width)
        
        path_slice = data[:, start:end, :]
        
        # Signature of a segment is a single vector; add a dim for stacking
        sig = _get_signature(path_slice, depth, univariate, streamed=False)
        all_sigs.append(np.expand_dims(sig, axis=1))

    concatenated_sigs = np.concatenate(all_sigs, axis=1)
    return torch.from_numpy(concatenated_sigs).to(device)

def local_signature_tight(
    data: torch.Tensor, x: np.ndarray, num_windows: int, 
    depth: int, univariate: bool, device: torch.device
) -> torch.Tensor:
    """Computes signatures on sequential, non-overlapping windows."""
    return _compute_local_sigs(data, x, num_windows, depth, univariate, device, width=0)

def local_signature_wide(
    data: torch.Tensor, x: np.ndarray, num_windows: int, width: int, 
    depth: int, univariate: bool, device: torch.device
) -> torch.Tensor:
    """Computes signatures on sequential, overlapping windows of a given width."""
    return _compute_local_sigs(data, x, num_windows, depth, univariate, device, width=width)

# --- Main Dispatcher Function ---

def ComputeSignatures(
    inputs: torch.Tensor, 
    x: np.ndarray, 
    config: SignatureConfig, 
    device: torch.device
) -> torch.Tensor:
    """
    Computes and concatenates a set of signature transforms based on a config.
    """
    output: List[torch.Tensor] = []
    
    if config.global_backward:
        output.append(global_signature_backward(
            inputs, x, config.num_windows, config.sig_level, config.univariate, device
        ))
    if config.global_forward:
        output.append(global_signature_forward(
            inputs, x, config.num_windows, config.sig_level, config.univariate, device
        ))
    if config.local_tight:
        output.append(local_signature_tight(
            inputs, x, config.num_windows, config.sig_level, config.univariate, device
        ))
    if config.local_wide:
        output.append(local_signature_wide(
            inputs, x, config.num_windows, config.local_width, config.sig_level, config.univariate, device
        ))
        
    return torch.cat(output, dim=2).float()