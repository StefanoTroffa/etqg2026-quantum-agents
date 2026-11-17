"""
QuantumBoardTransformerNet — Non-Markovian Position + Player + Strategy Transformer
Feature Extractor & Policy-Value Network (GitHub Issue #2 by Chris Chalkias).

Implements Chris Chalkias's architecture proposal:
  - Position tokens P_{1..N} (N=9 for TiqTaqToe 3x3, N=18 for Cheqqers 6x6 dark squares)
  - 2 Player tokens (X / O or Self / Opponent)
  - Non-Markovian Entanglement & Strategy Context tokens that encode the prior quantum
    move graph, superposition entropy, and opponent strategy signature (breaking the
    non-Markovianity of raw per-square marginal probabilities).
  - Equivariant / token-anchored policy readout:
    * TiqTaqToe (in_dim=111, out_dim=46):
      - Actions 0..35 (quantum pairs (i, j)): symmetric pairwise readout from attended
        Position Tokens (P_i, P_j) + Strategy token.
      - Actions 36..44 (classical square i): per-token readout from attended P_i.
      - Action 45 (measurement 'M'): readout from Strategy + Player tokens.
    * Cheqqers (in_dim=72, out_dim=288):
      - Actions 0..71 (18 squares x 4 diagonal directions): anchored at attended P_i.
      - Actions 72..179 (18 squares x 6 split pairs): anchored at attended P_i.
      - Actions 180..287 (18 squares x 6 merge pairs): anchored at attended P_i.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import QuantumModel


def _build_ttt_pair_indices() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precomputes:
      - pair_u: (36,) first square index for actions 0..35
      - pair_v: (36,) second square index for actions 0..35
      - sq_incident_edge_idx: (9, 8) index into the 36 entanglement edges for each square i
    """
    pairs: List[Tuple[int, int]] = []
    pair_map = {}
    idx = 0
    for i in range(9):
        for j in range(i + 1, 9):
            pairs.append((i, j))
            pair_map[(i, j)] = idx
            pair_map[(j, i)] = idx
            idx += 1

    pair_u = torch.tensor([u for u, _ in pairs], dtype=torch.long)
    pair_v = torch.tensor([v for _, v in pairs], dtype=torch.long)

    incident = torch.zeros((9, 8), dtype=torch.long)
    for i in range(9):
        col = 0
        for j in range(9):
            if i == j:
                continue
            incident[i, col] = pair_map[(i, j)]
            col += 1

    return pair_u, pair_v, incident


class QuantumBoardTransformerNet(QuantumModel):
    """
    Transformer-based Policy-Value Network for Quantum TiqTaqToe (111 -> 46)
    and Quantum Cheqqers (72 -> 288), with generic fallback for arbitrary (in_dim, out_dim).
    """

    def __init__(
        self,
        in_dim: int = 111,
        hidden_dim: int = 64,
        out_dim: int = 46,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        wdl: bool = False,
    ) -> None:
        super().__init__(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_model = max(32, int(hidden_dim))
        # Ensure d_model is divisible by nhead
        if self.d_model % nhead != 0:
            self.d_model = ((self.d_model // nhead) + 1) * nhead
        self.nhead = nhead
        self.num_layers = num_layers
        self.wdl = wdl

        pair_u, pair_v, incident = _build_ttt_pair_indices()
        self.register_buffer("ttt_pair_u", pair_u, persistent=False)
        self.register_buffer("ttt_pair_v", pair_v, persistent=False)
        self.register_buffer("ttt_incident_edges", incident, persistent=False)

        if in_dim == 111 and out_dim == 46:
            self.mode = "tiqtaqtoe"
            # Each of the 9 position tokens gets:
            # 6 square state channels + 2 marginal probs + 8 incident entanglement edges = 16 features
            self.pos_proj = nn.Linear(16, self.d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, 9, self.d_model) * 0.02)

            # 2 Player tokens (Self, Opponent) from global turn + total board mass + classic/quantum counts
            self.player_proj = nn.Linear(8, self.d_model)
            self.player_embed = nn.Parameter(torch.randn(1, 2, self.d_model) * 0.02)

            # 2 Non-Markovian Strategy / Entanglement tokens from 36 entanglement edges + 3 global features
            self.non_markov_proj = nn.Sequential(
                nn.Linear(39, self.d_model * 2),
                nn.GELU(),
            )
            self.non_markov_embed = nn.Parameter(torch.randn(1, 2, self.d_model) * 0.02)

            # Structured Action Readout Heads
            self.classical_sq_head = nn.Linear(self.d_model * 2, 1)
            self.pair_action_head = nn.Sequential(
                nn.Linear(self.d_model * 3 + 1, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
            )
            self.measure_head = nn.Linear(self.d_model * 2, 1)
            self.residual_mlp = nn.Sequential(
                nn.Linear(111, self.d_model * 2),
                nn.GELU(),
                nn.Linear(self.d_model * 2, 46),
            )

        elif in_dim == 72 and out_dim == 288:
            self.mode = "cheqqers"
            # Each of the 18 dark squares has 4 channels + 2 quantum superposition entropy features = 6 features
            self.pos_proj = nn.Linear(6, self.d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, 18, self.d_model) * 0.02)

            # 2 Player tokens (Self, Opponent)
            self.player_proj = nn.Linear(6, self.d_model)
            self.player_embed = nn.Parameter(torch.randn(1, 2, self.d_model) * 0.02)

            # 2 Non-Markovian Quantum Coherence / Strategy tokens from global board distribution
            self.non_markov_proj = nn.Sequential(
                nn.Linear(72, self.d_model * 2),
                nn.GELU(),
            )
            self.non_markov_embed = nn.Parameter(torch.randn(1, 2, self.d_model) * 0.02)

            # Per-square anchored readout: 4 classical/jump + 6 split + 6 merge = 16 actions per square (18 * 16 = 288)
            self.sq_classical_head = nn.Linear(self.d_model * 2, 4)
            self.sq_split_head = nn.Linear(self.d_model * 2, 6)
            self.sq_merge_head = nn.Linear(self.d_model * 2, 6)
            self.residual_mlp = nn.Sequential(
                nn.Linear(72, self.d_model * 2),
                nn.GELU(),
                nn.Linear(self.d_model * 2, 288),
            )
        else:
            self.mode = "generic"
            self.num_tokens = 8
            self.token_proj = nn.Linear(in_dim, self.num_tokens * self.d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, self.d_model) * 0.02)
            self.generic_policy_head = nn.Linear(self.d_model * 2, out_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.final_norm = nn.LayerNorm(self.d_model)

        val_out_dim = 3 if self.wdl else 1
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, val_out_dim),
        )

    def _encode_tiqtaqtoe_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        sq_feats_6 = x[:, :54].view(bsz, 9, 6)
        adj_36 = x[:, 54:90]
        probs_2 = x[:, 90:108].view(bsz, 9, 2)
        global_3 = x[:, 108:111]

        # Gather the 8 incident entanglement edges for each square i in 0..8
        incident_edges = adj_36[:, self.ttt_incident_edges]  # (B, 9, 8)
        pos_in = torch.cat([sq_feats_6, probs_2, incident_edges], dim=-1)  # (B, 9, 16)
        pos_tokens = self.pos_proj(pos_in) + self.pos_embed  # (B, 9, d_model)

        # Build 2 Player tokens (Self, Opponent)
        self_mass = probs_2[:, :, 0].sum(dim=-1, keepdim=True) / 9.0
        opp_mass = probs_2[:, :, 1].sum(dim=-1, keepdim=True) / 9.0
        self_classic = sq_feats_6[:, :, 0].sum(dim=-1, keepdim=True) / 3.0
        opp_classic = sq_feats_6[:, :, 1].sum(dim=-1, keepdim=True) / 3.0
        self_spooky = sq_feats_6[:, :, 4].sum(dim=-1, keepdim=True) / 6.0
        opp_spooky = sq_feats_6[:, :, 5].sum(dim=-1, keepdim=True) / 6.0

        p_self_in = torch.cat([self_mass, opp_mass, self_classic, self_spooky, global_3, torch.ones_like(self_mass)], dim=-1)
        p_opp_in = torch.cat([opp_mass, self_mass, opp_classic, opp_spooky, -global_3, -torch.ones_like(self_mass)], dim=-1)
        player_in = torch.stack([p_self_in, p_opp_in], dim=1)  # (B, 2, 8)
        player_tokens = self.player_proj(player_in) + self.player_embed  # (B, 2, d_model)

        # Build 2 Non-Markovian Strategy / Entanglement tokens
        nm_in = torch.cat([adj_36, global_3], dim=-1)  # (B, 39)
        nm_tokens = self.non_markov_proj(nm_in).view(bsz, 2, self.d_model) + self.non_markov_embed

        tokens = torch.cat([pos_tokens, player_tokens, nm_tokens], dim=1)  # (B, 13, d_model)
        return tokens, adj_36

    def _encode_cheqqers_tokens(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        sq_4 = x.view(bsz, 18, 4)
        # Quantum superposition indicator p * (1 - p) breaks non-Markovian ambiguity between classical vs split pieces
        self_p = sq_4[:, :, 0] + sq_4[:, :, 1]
        opp_p = sq_4[:, :, 2] + sq_4[:, :, 3]
        self_coh = (self_p * (1.0 - torch.clamp(self_p, 0.0, 1.0))).unsqueeze(-1) * 4.0
        opp_coh = (opp_p * (1.0 - torch.clamp(opp_p, 0.0, 1.0))).unsqueeze(-1) * 4.0

        pos_in = torch.cat([sq_4, self_coh, opp_coh], dim=-1)  # (B, 18, 6)
        pos_tokens = self.pos_proj(pos_in) + self.pos_embed  # (B, 18, d_model)

        # 2 Player tokens (Self, Opponent)
        self_pawns = sq_4[:, :, 0].sum(dim=-1, keepdim=True) / 6.0
        self_kings = sq_4[:, :, 1].sum(dim=-1, keepdim=True) / 3.0
        opp_pawns = sq_4[:, :, 2].sum(dim=-1, keepdim=True) / 6.0
        opp_kings = sq_4[:, :, 3].sum(dim=-1, keepdim=True) / 3.0
        self_coh_sum = self_coh.squeeze(-1).sum(dim=-1, keepdim=True) / 6.0
        opp_coh_sum = opp_coh.squeeze(-1).sum(dim=-1, keepdim=True) / 6.0

        p_self_in = torch.cat([self_pawns, self_kings, self_coh_sum, opp_pawns, opp_kings, opp_coh_sum], dim=-1)
        p_opp_in = torch.cat([opp_pawns, opp_kings, opp_coh_sum, self_pawns, self_kings, self_coh_sum], dim=-1)
        player_in = torch.stack([p_self_in, p_opp_in], dim=1)  # (B, 2, 6)
        player_tokens = self.player_proj(player_in) + self.player_embed

        # 2 Non-Markovian Strategy / Coherence tokens
        nm_tokens = self.non_markov_proj(x).view(bsz, 2, self.d_model) + self.non_markov_embed
        return torch.cat([pos_tokens, player_tokens, nm_tokens], dim=1)  # (B, 22, d_model)

    def _forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        single_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True

        bsz = x.shape[0]
        if self.mode == "tiqtaqtoe":
            tokens, adj_36 = self._encode_tiqtaqtoe_tokens(x)
            h = self.final_norm(self.encoder(tokens))
            h_pos = h[:, :9, :]        # (B, 9, d_model)
            h_play = h[:, 9:11, :]     # (B, 2, d_model)
            h_nm = h[:, 11:13, :]      # (B, 2, d_model)

            ctx_global = (h_play[:, 0, :] + h_nm[:, 0, :]) * 0.5  # (B, d_model)
            ctx_expanded_9 = ctx_global.unsqueeze(1).expand(bsz, 9, self.d_model)

            # Classical moves 36..44
            classical_in = torch.cat([h_pos, ctx_expanded_9], dim=-1)
            logits_classical_9 = self.classical_sq_head(classical_in).squeeze(-1)  # (B, 9)

            # Quantum pair moves 0..35
            h_u = h_pos[:, self.ttt_pair_u, :]  # (B, 36, d_model)
            h_v = h_pos[:, self.ttt_pair_v, :]  # (B, 36, d_model)
            ctx_36 = h_nm[:, 0, :].unsqueeze(1).expand(bsz, 36, self.d_model)
            pair_in = torch.cat(
                [h_u + h_v, h_u * h_v, ctx_36, adj_36.unsqueeze(-1)],
                dim=-1,
            )
            logits_pairs_36 = self.pair_action_head(pair_in).squeeze(-1)  # (B, 36)

            # Measurement move 45
            meas_in = torch.cat([h_play[:, 0, :], h_nm[:, 1, :]], dim=-1)
            logits_meas_1 = self.measure_head(meas_in)  # (B, 1)

            policy_logits = torch.cat([logits_pairs_36, logits_classical_9, logits_meas_1], dim=-1)
            policy_logits = policy_logits + 0.35 * self.residual_mlp(x)

            val_feat = torch.cat([h_pos.mean(dim=1), h_play[:, 0, :], h_nm[:, 0, :]], dim=-1)

        elif self.mode == "cheqqers":
            tokens = self._encode_cheqqers_tokens(x)
            h = self.final_norm(self.encoder(tokens))
            h_pos = h[:, :18, :]       # (B, 18, d_model)
            h_play = h[:, 18:20, :]    # (B, 2, d_model)
            h_nm = h[:, 20:22, :]      # (B, 2, d_model)

            ctx_global = (h_play[:, 0, :] + h_nm[:, 0, :]) * 0.5
            ctx_18 = ctx_global.unsqueeze(1).expand(bsz, 18, self.d_model)
            sq_ctx = torch.cat([h_pos, ctx_18], dim=-1)  # (B, 18, 2 * d_model)

            logits_c72 = self.sq_classical_head(sq_ctx).reshape(bsz, 72)
            logits_s108 = self.sq_split_head(sq_ctx).reshape(bsz, 108)
            logits_m108 = self.sq_merge_head(sq_ctx).reshape(bsz, 108)
            policy_logits = torch.cat([logits_c72, logits_s108, logits_m108], dim=-1)
            policy_logits = policy_logits + 0.35 * self.residual_mlp(x)

            val_feat = torch.cat([h_pos.mean(dim=1), h_play[:, 0, :], h_nm[:, 0, :]], dim=-1)

        else:
            tokens = self.token_proj(x).view(bsz, self.num_tokens, self.d_model) + self.pos_embed
            h = self.final_norm(self.encoder(tokens))
            pooled = torch.cat([h.mean(dim=1), h[:, 0, :]], dim=-1)
            policy_logits = self.generic_policy_head(pooled)
            val_feat = torch.cat([h.mean(dim=1), h[:, 0, :], h[:, -1, :]], dim=-1)

        wdl_logits = None
        if self.wdl:
            wdl_logits = self.value_head(val_feat)
            probs = torch.softmax(wdl_logits, dim=-1)
            value = probs[:, 0] - probs[:, 2]
        else:
            value = torch.tanh(self.value_head(val_feat)).squeeze(-1)

        if single_input:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
        return policy_logits, value, wdl_logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value, _ = self._forward_all(x)
        return policy_logits, value
