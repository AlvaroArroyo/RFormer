import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


def swish(x: Tensor) -> Tensor:
    """Swish activation: x * sigmoid(x)."""
    return x * x.sigmoid()


ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu,
}


class Conv1D(nn.Module):
    """
    Pointwise (1x1) convolution via linear transformation.
    """
    def __init__(self, out_dim: int, rf: int, in_dim: int):
        super().__init__()
        if rf != 1:
            raise NotImplementedError
        self.w = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.size()
        x_flat = x.view(-1, x.size(-1))
        out = x_flat @ self.w + self.b
        return out.view(batch, seq_len, -1)


class LayerNorm(nn.Module):
    """OpenAI-style layer norm with epsilon inside sqrt."""
    def __init__(self, n_embd: int, e: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        return self.g * (x - mu) / torch.sqrt(sigma + self.e) + self.b


class Attention(nn.Module):
    """Multi-head self-attention with optional sparse masking."""
    def __init__(
        self,
        config: Any,
        n_head: int,
        n_embd: int,
        win_len: int,
        scale: bool,
        q_len: int,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = (self.head_dim ** -0.5) if scale else 1.0

        # Mask
        if config.sparse:
            mask = self.log_mask(win_len, config.sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len))
        self.register_buffer('mask_tri', mask.view(1, 1, win_len, win_len))


        self.head_dim = n_embd // n_head      
        proj_dim = self.head_dim * n_head   
        # QKV and output projections
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, q_len, bias=False)
        self.value  = Conv1D(proj_dim, 1, n_embd)     
        self.c_proj = Conv1D(n_embd,   1, proj_dim)    

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def log_mask(self, win_len: int, sub_len: int) -> Tensor:
        mask = torch.zeros(win_len, win_len)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask

    def row_mask(self, index: int, sub_len: int, win_len: int) -> Tensor:
        mask = torch.zeros(win_len)
        log_l = math.ceil(math.log2(sub_len))
        if (win_len // sub_len) * 2 * log_l > index:
            mask[: index + 1] = 1
        else:
            i = index
            while i >= 0:
                start = i - log_l + 1
                if start < 0:
                    mask[:i] = 1
                    break
                mask[start : i + 1] = 1
                for j in range(log_l):
                    ni = start - 2**j
                    if ni >= 0 and (i - ni) <= sub_len:
                        mask[ni] = 1
                i -= sub_len
        return mask

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.size()
        # Value
        v = self.value(x).view(b, t, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Query & Key
        pad = (self.query_key.kernel_size[0] - 1, 0)
        x_pad = nn.functional.pad(x.transpose(1, 2), pad)
        qk = self.query_key(x_pad).permute(0, 2, 1)
        split = n_embd = qk.size(-1) // 2 // self.n_head  # recalculated head dim
        q, k = qk.split(self.n_head * split * 1, dim=2)
        q = q.view(b, t, self.n_head, split).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_head, split).permute(0, 2, 3, 1)

        # Attention scores
        scores = (q @ k) * self.scale
        mask = self.mask_tri[:, :, :t, :t]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum
        context = attn @ v
        context = context.permute(0, 2, 1, 3).contiguous().view(b, t, -1)

        out = self.c_proj(context)
        return self.resid_dropout(out)


class MLP(nn.Module):
    """Feed-forward network: two 1x1 convs with activation + dropout."""
    def __init__(
        self,
        config: Any,
        n_state: int,
        n_embd: int,
        acf = 'relu'
    ):
        super().__init__()
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = ACT_FNS[acf]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    """Single Transformer block: Attn -> AddNorm -> MLP -> AddNorm."""
    def __init__(
        self,
        config: Any,
        n_head: int,
        win_len: int,
        n_embd: int,
        scale: bool,
        q_len: int,
    ):
        super().__init__()
        self.attn = Attention(config, n_head, n_embd, win_len, scale, q_len)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(config, 4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = self.ln_1(x)
        x = x + self.mlp(x)
        return self.ln_2(x)


class TransformerModel(nn.Module):
    """Transformer model with config, same signature as before."""
    def __init__(
        self,
        config: Any,
        input_dim: int,
        n_head: int,
        seq_num: int,
        layer: int,
        n_embd: int,
        win_len: int,
    ):
        super().__init__()
        self.po_embed = nn.Embedding(win_len, n_embd)
        self.drop_em = nn.Dropout(config.embd_pdrop)
        base = Block(config, n_head, win_len, n_embd + input_dim, config.scale_att, config.q_len)
        self.blocks = nn.ModuleList([base for _ in range(layer)])

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.size()
        pos = torch.arange(t, device=x.device)
        pe = self.po_embed(pos).unsqueeze(0).expand(b, t, -1)
        x = torch.cat((x, pe), dim=2)
        for block in self.blocks:
            x = block(x)
        return x


class DecoderTransformer(nn.Module):
    """Decoder with same signature, using config."""
    def __init__(
        self,
        config: Any,
        input_dim: int,
        n_head: int,
        seq_num: int,
        layer: int,
        n_embd: int,
        win_len: int,
        num_classes: int,
    ):
        super().__init__()
        self.transformer = TransformerModel(config, input_dim, n_head, seq_num, layer, n_embd, win_len)
        self.mlp = nn.Linear(input_dim + n_embd, num_classes)
        self.final = nn.Linear(num_classes, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.transformer(x).mean(dim=1)
        return self.mlp(h)


class GaussianLoss(nn.Module):
    """Negative log-likelihood for Gaussian."""
    def __init__(self, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        loss = -Normal(self.mu, self.sigma).log_prob(x)
        return loss.mean()