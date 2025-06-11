# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch optimization for BERT model.
This file contains a cleaned version of the BERTAdam optimizer, preserving
the original's specific scheduling and weight decay behavior.
"""

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from typing import Callable, Dict, Iterable, List, Optional

# --- Learning Rate Schedulers with Warmup ---

def warmup_cosine(x: float, warmup: float = 0.002) -> float:
    """Linear warmup and cosine decay schedule."""
    if x < warmup:
        return x / warmup
    # Note: This follows the original implementation. A more common formulation
    # for the cosine decay part would be over the range [warmup, 1.0].
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x: float, warmup: float = 0.002) -> float:
    """Linear warmup and constant schedule."""
    if x < warmup:
        return x / warmup
    return 1.0

def warmup_linear(x: float, warmup: float = 0.002) -> float:
    """Linear warmup and linear decay schedule."""
    if x < warmup:
        return x / warmup
    # Note: This follows the original implementation. A more common formulation
    # for linear decay would be (1.0 - x) / (1.0 - warmup).
    return 1.0 - x

SCHEDULES: Dict[str, Callable] = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}

# --- BERTAdam Optimizer ---

class BERTAdam(Optimizer):
    """
    Implements the BERT version of the Adam algorithm with weight decay fix,
    warmup, and learning rate scheduling. This implementation is designed to
    match the original BERT paper's optimizer behavior.

    Args:
        params (Iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            The learning rate.
        warmup (float, optional, defaults to -1):
            Portion of t_total for the warmup, -1 means no warmup.
            Value should be in [0.0, 1.0).
        t_total (int, optional, defaults to -1):
            Total number of training steps for the learning rate schedule,
            -1 means constant learning rate.
        schedule (str, optional, defaults to 'warmup_linear'):
            Schedule to use for the warmup. Valid options are listed in SCHEDULES.
        b1 (float, optional, defaults to 0.9):
            Adam's beta1 parameter.
        b2 (float, optional, defaults to 0.999):
            Adam's beta2 parameter.
        e (float, optional, defaults to 1e-6):
            Adam's epsilon parameter for numerical stability.
        weight_decay_rate (float, optional, defaults to 0.01):
            Weight decay rate for the AdamW fix.
        max_grad_norm (float, optional, defaults to 1.0):
            Maximum norm for gradient clipping (-1 means no clipping).
    """
    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 warmup: float = -1.0,
                 t_total: int = -1,
                 schedule: str = 'warmup_linear',
                 b1: float = 0.9,
                 b2: float = 0.999,
                 e: float = 1e-6,
                 weight_decay_rate: float = 0.01,
                 max_grad_norm: float = 1.0):

        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if schedule not in SCHEDULES:
            raise ValueError(f"Invalid schedule parameter: {schedule}")
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(f"Invalid warmup: {warmup} - should be in [0.0, 1.0) or -1")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid b1 parameter: {b1} - should be in [0.0, 1.0)")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid b2 parameter: {b2} - should be in [0.0, 1.0)")
        if not e >= 0.0:
            raise ValueError(f"Invalid epsilon value: {e} - should be >= 0.0")

        defaults = dict(
            lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
            b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
            max_grad_norm=max_grad_norm
        )
        super().__init__(params, defaults)

    def get_lr(self) -> List[float]:
        """
        Calculates the current learning rate for each parameter.
        Note: This returns a list of LRs, one per parameter, not per group,
        and has unusual early exit behavior, which is preserved from the original.
        """
        lr_list = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Original behavior: return [0] if any parameter is not initialized.
                if len(state) == 0:
                    return [0.0]

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr_list.append(lr_scheduled)
        return lr_list

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BERTAdam does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Preserving per-parameter gradient clipping from the original implementation.
                # Note: The standard practice is to clip the norm of all gradients
                # in a group collectively.
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # AdamW update:
                # Decay the first and second moment running average coefficients.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = exp_avg / (exp_avg_sq.sqrt() + group['e'])

                # Apply weight decay as in AdamW.
                if group['weight_decay_rate'] > 0.0:
                    update.add_(p.data, alpha=group['weight_decay_rate'])

                # Calculate scheduled learning rate.
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                p.data.add_(update, alpha=-lr_scheduled)

                # Preserving per-parameter step increment from the original implementation.
                # Note: This is unconventional and causes the step count to advance
                # much faster than the number of optimizer steps.
                state['step'] += 1

        return loss