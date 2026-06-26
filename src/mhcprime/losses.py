import torch
import torch.nn as nn
import torch.nn.functional as F

# convert to smooth AP by doing 1 - softap.
class LogSmoothAPScore(nn.Module):
    def __init__(self, alpha=10.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, scores, labels):
        y = labels.float()
        B = y.size(0)
        P = y.sum().clamp(min=1.0)
        diff = scores.unsqueeze(0) - scores.unsqueeze(1)
        S = torch.sigmoid(self.alpha * diff)
        num = S.matmul(y)
        den = S.sum(dim=1).clamp(min=self.epsilon)
        soft_prec = num / den
        softap = (soft_prec * y).sum() / P
        return torch.log(1.0 / (softap + self.epsilon))

# spearman loss

# differentiable sorting
def soft_rank(x: torch.Tensor, tau: float = 1.0):
    """
    Differentiable rank based on pairwise sigmoid.
    r_i = 1 + Σ_j σ((x_j - x_i) / τ)
    τ -> 0  gives hard ranks.
    x : (..., L)
    returns  (..., L)  in [1, L]
    """
    x_col = x.unsqueeze(-1) # (..., L, 1)
    x_row = x.unsqueeze(-2) # (..., 1, L)
    P = torch.sigmoid((x_row - x_col) / tau) # pairwise prob
    return 1 + P.sum(-1)

# soft spearman loss
class SoftSpearmanLoss(nn.Module):
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # ensure 2-D
        if pred.dim() == 1:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        r_pred   = soft_rank(pred,   tau=self.tau)   # (B,L)
        r_target = soft_rank(target, tau=1e-4)       # ~hard ranks

        # centre
        r_pred   = r_pred   - r_pred.mean(-1, keepdim=True)
        r_target = r_target - r_target.mean(-1, keepdim=True)

        num = (r_pred * r_target).sum(-1)
        den = torch.norm(r_pred,   dim=-1) * torch.norm(r_target, dim=-1) + 1e-8
        rho = num / den

        return (1 - rho).mean()
