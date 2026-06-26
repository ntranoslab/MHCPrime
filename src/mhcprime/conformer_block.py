import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv1d(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.padding = kernel_size // 2
        self.dw = nn.Conv1d(dim, dim, kernel_size,
                            padding=self.padding, groups=dim, bias=True)

    def forward(self, x):
        # x: (B, L, D) → (B, D, L)
        return self.dw(x.transpose(1, 2)).transpose(1, 2)

class SwiGLUBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.inp = nn.Linear(d_model, 2 * d_ff) # replaces Linear(d→d_ff)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_ff, d_model)
        self.out_drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.ln(x)
        z = self.inp(x) # [B,L,2M]
        a, b = z.chunk(2, dim=-1) # [B,L,M], [B,L,M]
        y = a * F.silu(b) # SwiGLU
        y = self.drop(y)
        y = self.out(y)
        return self.out_drop(y)

# new conformer block w/ masked bos
class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 400,
        n_heads: int = 16,
        d_ff: int = None,
        conv_kernel: int = 5,
        dropout: float = 0.1,
        mask_bos: bool = False,
        n_start_pos_to_mask: int = 0, # for masking the first n positions.
        ff_pre_post_swiglu: bool = False
    ):
        super().__init__()
        self.mask_bos = mask_bos # not used, but left in here for backward compatability.

        self.n_start_pos_to_mask = n_start_pos_to_mask

        if d_ff is None:
            d_ff = 4 * d_model
        
        if ff_pre_post_swiglu:
            print("Using SwiGLU in conformer")

            # adjust d_ff for param parity with gelu
            d_ff = int(round(8 * d_model / 3))   

            # use canonical swiglu, not the MxM variant
            self.ff_pre  = SwiGLUBlock(d_model, d_ff, dropout)
            self.ff_post = SwiGLUBlock(d_model, d_ff, dropout)

        else:
            self.ff_pre = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.ff_post = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        # MHSA
        self.attn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        # Conv branch
        self.conv_ln = nn.LayerNorm(d_model)
        self.conv_pw1 = nn.Linear(d_model, 2 * d_model)
        self.conv_dw = DepthwiseConv1d(d_model, conv_kernel)
        self.conv_bn = nn.BatchNorm1d(d_model)
        self.conv_pw2 = nn.Linear(d_model, d_model)
        self.conv_drop = nn.Dropout(dropout)

        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor, # (B,L,D)
        mask: torch.Tensor = None, # True = keep
        output_attentions: bool = False,
    ):
        x = x + 0.5 * self.ff_pre(x)

        attn_mask = ~mask if mask is not None else None
        qkv = self.attn_ln(x)
        attn_out, attn_w = self.self_attn(
            qkv, qkv, qkv,
            key_padding_mask=attn_mask,
            need_weights=output_attentions,
        )
        x = x + self.attn_drop(attn_out)

        conv_mask = mask
        if (mask is not None) and (self.n_start_pos_to_mask > 0):
            conv_mask = mask.clone()
            n = min(self.n_start_pos_to_mask, conv_mask.size(1))
            if n > 0:
                conv_mask[:, :n] = False

        y = self.conv_ln(x)
        y = self.conv_pw1(y)
        a, b = y.chunk(2, dim=-1)
        y = a * torch.sigmoid(b) # GLU
        y = self.conv_dw(y)
        y = self.conv_bn(y.transpose(1, 2)).transpose(1, 2)
        y = F.silu(y)
        y = self.conv_pw2(y)
        if conv_mask is not None:
            y = y.masked_fill(~conv_mask.unsqueeze(-1), 0.)
        x = x + self.conv_drop(y)

        x = x + 0.5 * self.ff_post(x)
        x = self.final_ln(x)

        return (x, attn_w) if output_attentions else x


class ConformerEncoder(nn.Module):
    def __init__(self, num_layers: int, **conf_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [ConformerBlock(**conf_kwargs) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        attn_list = []
        for layer in self.layers:
            if output_attentions:
                x, attn_w = layer(x, mask=mask, output_attentions=True)
                attn_list.append(attn_w)
            else:
                x = layer(x, mask=mask)
        return (x, attn_list) if output_attentions else x

