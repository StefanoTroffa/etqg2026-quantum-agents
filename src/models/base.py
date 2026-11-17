import torch
import torch.nn as nn

class QuantumModel(nn.Module):
    """
    Abstract interface for all models in the quantum games reinforcement learning pipeline.
    Inherits from PyTorch's nn.Module.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass.
        Args:
            x: Input tensor representing the environment state.
        Returns:
            Output tensor of Q-values for each possible action.
        """
        raise NotImplementedError
