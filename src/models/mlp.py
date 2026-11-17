import torch
import torch.nn as nn

from models.base import QuantumModel

class MLPNet(QuantumModel):
    """
    Multi-Layer Perceptron (MLP) for Actor-Critic (Policy-Value) learning.
    Supports dynamic layer sizing, depth, decoupled backbones, and WDL values.
    """
    def __init__(self, in_dim=111, hidden_dim=64, out_dim=46, hidden_dims=None, decoupled=False, wdl=False):
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.decoupled = decoupled
        self.wdl = wdl
        
        if hidden_dims is None:
            if isinstance(hidden_dim, list):
                self.hidden_dims = hidden_dim
            else:
                self.hidden_dims = [hidden_dim, hidden_dim]
        else:
            self.hidden_dims = hidden_dims

        if self.decoupled:
            # Separate backbone for policy path
            policy_backbone_layers = []
            prev_dim = in_dim
            for h_dim in self.hidden_dims:
                policy_backbone_layers.append(nn.Linear(prev_dim, h_dim))
                policy_backbone_layers.append(nn.ReLU())
                prev_dim = h_dim
            self.policy_backbone = nn.Sequential(*policy_backbone_layers)
            
            # Separate backbone for value path
            value_backbone_layers = []
            prev_dim = in_dim
            for h_dim in self.hidden_dims:
                value_backbone_layers.append(nn.Linear(prev_dim, h_dim))
                value_backbone_layers.append(nn.ReLU())
                prev_dim = h_dim
            self.value_backbone = nn.Sequential(*value_backbone_layers)
        else:
            backbone_layers = []
            prev_dim = in_dim
            for h_dim in self.hidden_dims:
                backbone_layers.append(nn.Linear(prev_dim, h_dim))
                backbone_layers.append(nn.ReLU())
                prev_dim = h_dim
            self.backbone = nn.Sequential(*backbone_layers)
            
        self.policy_head = nn.Linear(prev_dim, out_dim)
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
        out_layer_idx = 2 * len(self.hidden_dims)
        
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
                
        # If model expects WDL value head but state_dict contains single scalar value_head, 
        # load strict=False to avoid error for value_head mismatch
        is_wdl_mismatch = self.wdl and any(k.startswith("value_head.") for k in state_dict.keys()) and \
                           state_dict.get("value_head.weight") is not None and \
                           state_dict["value_head.weight"].shape[0] != 3
                           
        if has_old_keys or is_wdl_mismatch:
            super().load_state_dict(new_state_dict, strict=False)
        else:
            super().load_state_dict(new_state_dict, strict=strict)
