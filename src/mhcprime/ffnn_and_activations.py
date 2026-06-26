import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

# not actually swiglu, more like sigmoid gated GLU. Optinally change later to actual swiglu later if preferred.
class SwiGLU(nn.Module):
    """
    SwiGLU Activation: Applies gating to FFN output.
    Equivalent to (FFN(x) * Sigmoid(Gate(x)))
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)  # Learnable gate
        self.proj = nn.Linear(dim, dim)  # Main FFN projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * torch.sigmoid(self.gate(x))

class SiLU(nn.Module):
    """
    SiLU activation (a.k.a. Swish).
    SiLU(x) = x * sigmoid(x)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class GeGLU(nn.Module):
    """
    GeGLU activation with learnable linear transforms.
    GeGLU(x) = GELU(W_left x) * (W_right x)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.left = nn.Linear(dim, dim)
        self.right = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.left(x)) * self.right(x)

class Mish(nn.Module):
    """
    Mish activation.
    Mish(x) = x * tanh(softplus(x))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class ReLU(nn.Module):
    """
    Standard ReLU activation.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

class ReLUSquared(nn.Module):
    """
    ReLU-squared activation.
    ReLUSquared(x) = (ReLU(x))^2
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)

class StarReLU(nn.Module):
    """
    StarReLU activation.
    For x >= 0: x
    For x < 0 : alpha * (exp(x) - 1)
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.alpha * (torch.exp(x) - 1))


class FFNNHead(nn.Module):
    def __init__(self, input_dim: int = 400, hidden_dims: list = [256, 128], dropout: float = 0.5, output_dim=1, pooling: str = "mean", activation: str = "swish"):
        super().__init__()
        self.pooling = pooling
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Dropout(dropout),
                SwiGLU(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.ffnn = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "max":
                x, _ = x.max(dim=1)
            elif self.pooling == "cls":
                x = x[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling}")
        
        return self.ffnn(x).squeeze(-1)

# with added layernorm
class FFNNHead_w_LayerNorm(nn.Module):
    def __init__(
        self,
        input_dim: int = 400,
        hidden_dims: list = [256, 128],
        dropout: float = 0.5,
        output_dim: int = 1,
        pooling: str = "mean",
        activation: str = "swish",
        use_layernorm: bool = False
    ):
        super().__init__()
        self.pooling = pooling
        self.use_layernorm = use_layernorm

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if self.use_layernorm:
                layers.append(nn.LayerNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(SwiGLU(hidden_dim))
            prev_dim = hidden_dim

        if self.use_layernorm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))

        self.ffnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # [batch, seq_len, feat]
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "max":
                x, _ = x.max(dim=1)
            elif self.pooling == "cls":
                x = x[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return self.ffnn(x).squeeze(-1)

"""
Legacy modules.
"""

class ESMLanguageModelingHead(nn.Module):
    """
    Language modeling head following ESM-style architecture.
    """
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        
    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = torch.nn.functional.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class AttentionPoolingHead(nn.Module):
    """
    Attention pooling head for sequence classification tasks.
    Uses attention mechanism to compute a weighted sum of sequence representations.
    """
    def __init__(
        self, 
        input_dim: int = 400, 
        hidden_dims: list = [256, 128], 
        dropout: float = 0.5, 
        output_dim: int = 1,
        attention_dim: int = 64,
        attention_dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation_fn: str = "gelu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.attention_query = nn.Linear(input_dim, attention_dim)
        self.attention_key = nn.Linear(input_dim, attention_dim)
        self.attention_value = nn.Linear(input_dim, input_dim)
        self.scale = math.sqrt(attention_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        
        if activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "swish":
            self.activation = Swish()
        elif activation_fn == "mish":
            self.activation = Mish()
        elif activation_fn == "swiglu":
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if activation_fn == "swiglu":
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.Dropout(dropout))
                layers.append(SwiGLU(hidden_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Dropout(dropout),
                    self.activation
                ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.ffnn = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        query = self.attention_query(x) # [batch_size, seq_len, attention_dim]
        key = self.attention_key(x) # [batch_size, seq_len, attention_dim]
        value = self.attention_value(x) # [batch_size, seq_len, input_dim]
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale # [batch_size, seq_len, seq_len]
        
        if padding_mask is not None:
            attention_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1) # [batch_size, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask, -10000.0)
        
        attention_weights = torch.softmax(attention_scores, dim=-1) # [batch_size, seq_len, seq_len]
        attention_weights = self.attention_dropout(attention_weights)
        weighted_sum = torch.matmul(attention_weights, value) # [batch_size, seq_len, input_dim]
        pooled = weighted_sum.mean(dim=1) # [batch_size, input_dim]
        output = self.ffnn(pooled) # [batch_size, output_dim]
        
        return output.squeeze(-1) if self.output_dim == 1 else output
