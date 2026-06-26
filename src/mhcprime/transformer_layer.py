import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int = 400, n_heads: int = 16, dff_mp: int=1, dropout=0.0):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dff_mp * d_model,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        # Access the self-attention module directly
        self.self_attn = self.transformer_layer.self_attn
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, output_attentions: bool = False) -> torch.Tensor:
        """
        Forward pass for transformer layer with optional attention weight output.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to output attention weights
            
        Returns:
            Tensor or tuple of (tensor, attention_weights)
        """
        if mask is not None:
            # Convert boolean mask (True=keep, False=ignore) to float attention mask
            attention_mask = mask.to(dtype=torch.bool)  # Ensure boolean type
        else:
            attention_mask = None
            
        if output_attentions:
            # When we need attention weights, manually apply the transformer steps
            # to capture the attention weights
            src = x
            src_key_padding_mask = attention_mask
            
            # Get attention weights using the self-attention module
            # This returns attn_output, attn_weights
            src2, attn_weights = self.self_attn(
                src, src, src,
                attn_mask=None,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            
            # Apply the rest of the transformer layer operations manually
            src = src + self.transformer_layer.dropout1(src2)
            src = self.transformer_layer.norm1(src)
            
            # FFN
            src2 = self.transformer_layer.linear2(self.transformer_layer.dropout(
                self.transformer_layer.activation(self.transformer_layer.linear1(src))
            ))
            src = src + self.transformer_layer.dropout2(src2)
            src = self.transformer_layer.norm2(src)
            
            return src, attn_weights
        else:
            # Standard forward pass without returning attention weights
            return self.transformer_layer(x, src_key_padding_mask=attention_mask)

