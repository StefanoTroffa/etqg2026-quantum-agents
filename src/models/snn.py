import torch
import torch.nn as nn
import numpy as np

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid Surrogate Gradient for Spiking Neurons.
    """
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        # Derivative of Fast Sigmoid: 1 / (1 + alpha * |x|)^2
        grad = grad_output / (1.0 + alpha * torch.abs(x)).pow(2)
        return grad, None

# Alias for ease of use
spike_fn = FastSigmoidSurrogate.apply

class LIFCell(nn.Module):
    """
    A single Leaky Integrate-and-Fire neuron layer.
    """
    def __init__(self, in_features, out_features, beta=0.85, threshold=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x_spikes, mem, state_dict=None):
        """
        Args:
            x_spikes: Spikes from previous layer (shape: [batch_size, in_features])
            mem: Previous membrane potential (shape: [batch_size, out_features])
        """
        # Synaptic current
        syn = self.linear(x_spikes)
        
        # Leaky integrate
        mem_next = self.beta * mem + syn
        
        # Fire spike
        spikes = spike_fn(mem_next - self.threshold)
        
        # Reset membrane potential (subtraction method preserves residual potential)
        mem_next = mem_next - spikes * self.threshold
        
        return spikes, mem_next


from models.base import QuantumModel

class SpikingNet(QuantumModel):
    """
    Multi-layer Spiking Neural Network designed for Actor-Critic (Policy-Value) learning.
    """
    def __init__(self, in_dim=27, hidden_dim=64, out_dim=9, steps=20, beta=0.85):
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.steps = steps  # Simulation time steps (T)
        self.beta = beta

        # Layers: Spiking hidden layers, non-spiking linear readout layer
        self.fc1 = LIFCell(in_dim, hidden_dim, beta=beta)
        self.fc2 = LIFCell(hidden_dim, hidden_dim, beta=beta)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Continuous input features (shape: [batch_size, in_dim])
        Returns:
            Tuple of (policy_logits, value_scalar)
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize membrane potentials for spiking layers
        mem1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        mem2 = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Accumulator for Hidden Layer 2 spikes
        spk2_sum = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Precompute input spikes for all steps
        x_clipped = torch.clamp(x, 0.0, 1.0)
        if self.training:
            # Random Poisson spikes: shape [steps, batch_size, in_dim]
            in_spikes_seq = (torch.rand(self.steps, batch_size, self.in_dim, device=device) < x_clipped.unsqueeze(0)).float()
        else:
            # Deterministic rate encoding using Golden Ratio Weyl sequence
            thresholds = torch.tensor([(t * 0.6180339887498949) % 1.0 for t in range(self.steps)], device=device).view(self.steps, 1, 1)
            in_spikes_seq = (x_clipped.unsqueeze(0) > thresholds).float()

        # Simulation loop over time steps T
        for t in range(self.steps):
            # 2. Hidden Layer 1
            spk1, mem1 = self.fc1(in_spikes_seq[t], mem1)

            # 3. Hidden Layer 2
            spk2, mem2 = self.fc2(spk1, mem2)

            # Accumulate hidden spikes
            spk2_sum += spk2

        # 4. Continuous Linear Readout from average hidden spikes
        avg_spk2 = spk2_sum / self.steps
        policy_logits = self.fc3(avg_spk2)
        value = torch.tanh(self.fc4(avg_spk2)).squeeze(-1)
        
        return policy_logits, value

    def load_state_dict(self, state_dict, strict=True):
        has_fc4 = any(k.startswith("fc4.") for k in state_dict.keys())
        if not has_fc4:
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=strict)
