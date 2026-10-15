"""Two-layer Vanilla GCN implemented directly with PyTorch (no PyG GCNConv)."""
import torch
from torch import nn


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """A_hat = D_tilde^(-1/2) (A + I) D_tilde^(-1/2)."""
    n = A.shape[0]
    A_tilde = A + torch.eye(n, dtype=A.dtype, device=A.device)
    degrees = A_tilde.sum(dim=1)
    inv_sqrt = torch.pow(degrees, -0.5)
    D_inv_sqrt = torch.diag(inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


class VanillaGCN(nn.Module):
    """Node classifier: H1 = ReLU(A_hat X W0), Z = A_hat H1 W1."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = A_hat @ self.fc1(X)
        H = torch.relu(H)
        H = self.dropout(H)
        logits = A_hat @ self.fc2(H)
        return logits
