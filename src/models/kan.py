import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) linear layer using Radial Basis Functions (RBF) 
    as the learnable 1D activation functions on edges.
    
    Formula for each edge connection:
    phi(x) = w_base * SiLU(x) + sum( w_spline_k * exp( -((x - mu_k) / sigma)^2 ) )
    """
    def __init__(self, in_features, out_features, num_centers=8, grid_min=-1.5, grid_max=1.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        
        # 1. Base activation (residual connection)
        self.base_activation = nn.SiLU()
        self.weight_base = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_base, a=1.0)
        
        # 2. RBF learnable spline weight
        self.weight_spline = nn.Parameter(torch.Tensor(out_features, in_features * num_centers))
        nn.init.normal_(self.weight_spline, mean=0.0, std=0.1 / (in_features ** 0.5))
        
        # 3. Centers (grid points)
        centers = torch.linspace(grid_min, grid_max, num_centers)
        self.register_buffer("centers", centers)
        
        # RBF grid width (sigma)
        self.sigma = (grid_max - grid_min) / (num_centers - 1)
        
    def forward(self, x):
        # x shape: [batch_size, in_features]
        
        # Base output: [batch_size, out_features]
        base_output = F.linear(self.base_activation(x), self.weight_base)
        
        # RBF evaluation:
        # unsqueeze x to shape [batch_size, in_features, 1]
        # self.centers shape [num_centers]
        diff = x.unsqueeze(-1) - self.centers.view(1, 1, -1)
        rbf = torch.exp(-(diff / self.sigma).pow(2)) # [batch_size, in_features, num_centers]
        
        # Flatten RBF to [batch_size, in_features * num_centers]
        rbf_flat = rbf.view(x.size(0), -1)
        
        # Spline output: [batch_size, out_features]
        spline_output = F.linear(rbf_flat, self.weight_spline)
        
        return base_output + spline_output


from models.base import QuantumModel

class KANNet(QuantumModel):
    """
    Kolmogorov-Arnold Network (KAN) for Actor-Critic (Policy-Value) learning.
    Replaces traditional Linear layers and ReLU activations with KANLinear layers.
    Supports decoupled backbones and categorical WDL value heads.
    """
    def __init__(self, in_dim=111, hidden_dims=[64, 64], out_dim=46, num_centers=8, decoupled=False, wdl=False):
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.num_centers = num_centers
        self.decoupled = decoupled
        self.wdl = wdl
        
        if self.decoupled:
            # Separate KAN backbone for policy path
            policy_backbone_layers = []
            prev_dim = in_dim
            for h_dim in hidden_dims:
                policy_backbone_layers.append(KANLinear(prev_dim, h_dim, num_centers=num_centers))
                prev_dim = h_dim
            self.policy_backbone = nn.Sequential(*policy_backbone_layers)
            
            # Separate KAN backbone for value path
            value_backbone_layers = []
            prev_dim = in_dim
            for h_dim in hidden_dims:
                value_backbone_layers.append(KANLinear(prev_dim, h_dim, num_centers=num_centers))
                prev_dim = h_dim
            self.value_backbone = nn.Sequential(*value_backbone_layers)
        else:
            backbone_layers = []
            prev_dim = in_dim
            for h_dim in hidden_dims:
                backbone_layers.append(KANLinear(prev_dim, h_dim, num_centers=num_centers))
                prev_dim = h_dim
            self.backbone = nn.Sequential(*backbone_layers)
            
        self.policy_head = KANLinear(prev_dim, out_dim, num_centers=num_centers)
        if self.wdl:
            self.value_head = nn.Linear(prev_dim, 3) # Win, Draw, Loss logits
        else:
            self.value_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        policy_logits, value, _ = self._forward_all(x)
        return policy_logits, value

    def _forward_all(self, x):
        if self.decoupled:
            policy_features = self.policy_backbone(x)
            value_features = self.value_backbone(x)
        else:
            policy_features = self.backbone(x)
            value_features = policy_features
            
        policy_logits = self.policy_head(policy_features)
        
        wdl_logits = None
        if self.wdl:
            wdl_logits = self.value_head(value_features)
            probs = torch.softmax(wdl_logits, dim=-1)
            # V = P(win) - P(loss)
            value = probs[:, 0] - probs[:, 2]
        else:
            value = torch.tanh(self.value_head(value_features)).squeeze(-1)
            
        return policy_logits, value, wdl_logits

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        has_old_keys = False
        out_layer_idx = len(self.hidden_dims)
        
        for k, v in state_dict.items():
            if k.startswith("network."):
                has_old_keys = True
                parts = k.split(".")
                idx = int(parts[1])
                suffix = ".".join(parts[2:])
                
                if idx < out_layer_idx:
                    if self.decoupled:
                        new_state_dict[f"policy_backbone.{idx}.{suffix}"] = v
                        new_state_dict[f"value_backbone.{idx}.{suffix}"] = v
                    else:
                        new_state_dict[f"backbone.{idx}.{suffix}"] = v
                else:
                    new_state_dict[f"policy_head.{suffix}"] = v
            elif k.startswith("backbone."):
                if self.decoupled:
                    parts = k.split(".")
                    suffix = ".".join(parts[1:])
                    new_state_dict[f"policy_backbone.{suffix}"] = v
                    new_state_dict[f"value_backbone.{suffix}"] = v
                else:
                    new_state_dict[k] = v
            else:
                new_state_dict[k] = v
                
        is_wdl_mismatch = self.wdl and any(k.startswith("value_head.") for k in state_dict.keys()) and \
                           state_dict.get("value_head.weight") is not None and \
                           state_dict["value_head.weight"].shape[0] != 3
                           
        if has_old_keys or is_wdl_mismatch:
            super().load_state_dict(new_state_dict, strict=False)
        else:
            super().load_state_dict(new_state_dict, strict=strict)
