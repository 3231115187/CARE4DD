# -*- coding: utf-8 -*-
"""
care4dd_hgconv_model.py

A lightweight relation-aware heterogeneous graph convolution enhanced CARE-4DD model.

Pipeline:
1) Type-specific projection: X_p, X_d, X_r -> H_p, H_d, H_r.
2) Lightweight heterogeneous graph convolution:
   Patient receives Drug/Procedure messages; Drug/Procedure receive Patient messages.
3) PPR Top-K indices select evidence nodes:
   Z_d = H_d_tilde[topk_drug_ids], Z_r = H_r_tilde[topk_proc_ids].
4) Evidence patch construction:
   [Patient token | Drug evidence tokens | Procedure evidence tokens].
5) Patch encoder: token mixing + channel mixing.
6) Diagnosis prediction.

PPR decides "which nodes are selected".
The heterogeneous graph convolution decides "what representation the selected nodes carry".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_index_select(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Index select with support for -1 padding indices."""
    if index.dim() != 2:
        raise ValueError(f"top-k index should be [B, K], got {tuple(index.shape)}")
    pad_mask = index < 0
    safe_index = index.clamp_min(0)
    out = x[safe_index]
    if pad_mask.any():
        out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    return out


def mean_aggregate(
    src_x: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    """Mean aggregation from source nodes to destination nodes."""
    if src_index.numel() == 0:
        return src_x.new_zeros((num_dst, src_x.size(-1)))

    src_index = src_index.long()
    dst_index = dst_index.long()

    valid = (
        (src_index >= 0) & (src_index < src_x.size(0)) &
        (dst_index >= 0) & (dst_index < num_dst)
    )
    if valid.sum() == 0:
        return src_x.new_zeros((num_dst, src_x.size(-1)))

    src_index = src_index[valid]
    dst_index = dst_index[valid]

    msg = src_x[src_index]
    out = src_x.new_zeros((num_dst, src_x.size(-1)))
    out.index_add_(0, dst_index, msg)

    deg = src_x.new_zeros((num_dst, 1))
    deg.index_add_(
        0,
        dst_index,
        torch.ones((dst_index.numel(), 1), device=src_x.device, dtype=src_x.dtype),
    )
    out = out / deg.clamp_min(1.0)
    return out


class TypeSpecificProjection(nn.Module):
    """Project Patient / Drug / Procedure raw features into a shared hidden space."""

    def __init__(
        self,
        patient_in_dim: int,
        drug_in_dim: int,
        proc_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patient_proj = nn.Sequential(
            nn.Linear(patient_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proc_proj = nn.Sequential(
            nn.Linear(proc_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x_patient: torch.Tensor,
        x_drug: torch.Tensor,
        x_proc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.patient_proj(x_patient), self.drug_proj(x_drug), self.proc_proj(x_proc)


class LightRelationHeteroGraphConv(nn.Module):
    """A lightweight relation-aware heterogeneous graph convolution.

    Relation directions:
    - Drug -> Patient
    - Procedure -> Patient
    - Patient -> Drug
    - Patient -> Procedure

    The input node representations must already share the same hidden dimension.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_relation_gate: bool = True,
    ) -> None:
        super().__init__()

        self.d2p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.r2p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.p2d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.p2r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.self_p = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.self_d = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.self_r = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_p = nn.LayerNorm(hidden_dim)
        self.norm_d = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.use_relation_gate = use_relation_gate

        if use_relation_gate:
            self.patient_rel_gate = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        h_patient: torch.Tensor,
        h_drug: torch.Tensor,
        h_proc: torch.Tensor,
        patient_drug_edge_index: torch.Tensor,
        patient_proc_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward.

        patient_drug_edge_index:
            [2, E_pd], row 0 = patient local id, row 1 = drug local id.
        patient_proc_edge_index:
            [2, E_pr], row 0 = patient local id, row 1 = procedure local id.
        """
        device = h_patient.device

        if patient_drug_edge_index is None or patient_drug_edge_index.numel() == 0:
            pd_p = torch.empty(0, dtype=torch.long, device=device)
            pd_d = torch.empty(0, dtype=torch.long, device=device)
        else:
            patient_drug_edge_index = patient_drug_edge_index.to(device).long()
            pd_p = patient_drug_edge_index[0]
            pd_d = patient_drug_edge_index[1]

        if patient_proc_edge_index is None or patient_proc_edge_index.numel() == 0:
            pr_p = torch.empty(0, dtype=torch.long, device=device)
            pr_r = torch.empty(0, dtype=torch.long, device=device)
        else:
            patient_proc_edge_index = patient_proc_edge_index.to(device).long()
            pr_p = patient_proc_edge_index[0]
            pr_r = patient_proc_edge_index[1]

        # Entity -> Patient
        drug_to_patient = mean_aggregate(
            src_x=h_drug,
            src_index=pd_d,
            dst_index=pd_p,
            num_dst=h_patient.size(0),
        )
        proc_to_patient = mean_aggregate(
            src_x=h_proc,
            src_index=pr_r,
            dst_index=pr_p,
            num_dst=h_patient.size(0),
        )
        drug_to_patient = self.d2p(drug_to_patient)
        proc_to_patient = self.r2p(proc_to_patient)

        if self.use_relation_gate:
            gate = torch.softmax(self.patient_rel_gate, dim=0)
            patient_msg = gate[0] * drug_to_patient + gate[1] * proc_to_patient
        else:
            patient_msg = 0.5 * (drug_to_patient + proc_to_patient)

        # Patient -> Drug
        patient_to_drug = mean_aggregate(
            src_x=h_patient,
            src_index=pd_p,
            dst_index=pd_d,
            num_dst=h_drug.size(0),
        )
        patient_to_drug = self.p2d(patient_to_drug)

        # Patient -> Procedure
        patient_to_proc = mean_aggregate(
            src_x=h_patient,
            src_index=pr_p,
            dst_index=pr_r,
            num_dst=h_proc.size(0),
        )
        patient_to_proc = self.p2r(patient_to_proc)

        new_patient = self.norm_p(h_patient + self.dropout(F.gelu(self.self_p(h_patient) + patient_msg)))
        new_drug = self.norm_d(h_drug + self.dropout(F.gelu(self.self_d(h_drug) + patient_to_drug)))
        new_proc = self.norm_r(h_proc + self.dropout(F.gelu(self.self_r(h_proc) + patient_to_proc)))

        return new_patient, new_drug, new_proc


class MixerBlock(nn.Module):
    """MLP-Mixer style patch encoder block."""

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        token_mlp_dim: Optional[int] = None,
        channel_mlp_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        token_mlp_dim = token_mlp_dim or max(16, num_tokens * 2)
        channel_mlp_dim = channel_mlp_dim or hidden_dim * 2

        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_tokens),
            nn.Dropout(dropout),
        )

        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing: [B, L, D] -> [B, D, L] -> [B, L, D]
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel mixing: [B, L, D]
        y = self.channel_norm(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


class PatchEncoder(nn.Module):
    """Stacked token/channel mixing encoder."""

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        token_mlp_dim: Optional[int] = None,
        channel_mlp_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(x)


@dataclass
class CARE4DDHGConvConfig:
    patient_in_dim: int
    drug_in_dim: int
    proc_in_dim: int
    num_classes: int

    hidden_dim: int = 64
    num_drug_tokens: int = 32
    num_proc_tokens: int = 31
    encoder_layers: int = 3
    token_mlp_dim: Optional[int] = None
    channel_mlp_dim: Optional[int] = None
    dropout: float = 0.1
    use_hgconv: bool = True
    use_type_embedding: bool = True


class CARE4DD_HGConv(nn.Module):
    """CARE-4DD with a lightweight heterogeneous graph convolution before PPR patch gathering.

    Expected forward inputs:
    - x_patient: [N_p, F_p]
    - x_drug: [N_d, F_d]
    - x_proc: [N_r, F_r]
    - patient_drug_edge_index: [2, E_pd], patient local id -> drug local id
    - patient_proc_edge_index: [2, E_pr], patient local id -> procedure local id
    - batch_patient_ids: [B]
    - topk_drug_ids: [B, K_d], PPR Top-K drug indices. Padding can be -1.
    - topk_proc_ids: [B, K_r], PPR Top-K procedure indices. Padding can be -1.
    """

    def __init__(self, cfg: CARE4DDHGConvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tokens = 1 + cfg.num_drug_tokens + cfg.num_proc_tokens

        self.proj = TypeSpecificProjection(
            patient_in_dim=cfg.patient_in_dim,
            drug_in_dim=cfg.drug_in_dim,
            proc_in_dim=cfg.proc_in_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        self.hgconv = LightRelationHeteroGraphConv(
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_relation_gate=True,
        )

        if cfg.use_type_embedding:
            self.type_embedding = nn.Embedding(3, cfg.hidden_dim)
        else:
            self.type_embedding = None

        self.patch_encoder = PatchEncoder(
            num_tokens=self.num_tokens,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.encoder_layers,
            token_mlp_dim=cfg.token_mlp_dim,
            channel_mlp_dim=cfg.channel_mlp_dim,
            dropout=cfg.dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def encode_all_nodes(
        self,
        x_patient: torch.Tensor,
        x_drug: torch.Tensor,
        x_proc: torch.Tensor,
        patient_drug_edge_index: torch.Tensor,
        patient_proc_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_patient, h_drug, h_proc = self.proj(x_patient, x_drug, x_proc)

        if self.cfg.use_hgconv:
            h_patient, h_drug, h_proc = self.hgconv(
                h_patient=h_patient,
                h_drug=h_drug,
                h_proc=h_proc,
                patient_drug_edge_index=patient_drug_edge_index,
                patient_proc_edge_index=patient_proc_edge_index,
            )
        return h_patient, h_drug, h_proc

    def build_patch(
        self,
        h_patient: torch.Tensor,
        h_drug: torch.Tensor,
        h_proc: torch.Tensor,
        batch_patient_ids: torch.Tensor,
        topk_drug_ids: torch.Tensor,
        topk_proc_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_patient_ids = batch_patient_ids.long().to(h_patient.device)
        topk_drug_ids = topk_drug_ids.long().to(h_patient.device)
        topk_proc_ids = topk_proc_ids.long().to(h_patient.device)

        patient_token = h_patient[batch_patient_ids].unsqueeze(1)
        drug_tokens = _safe_index_select(h_drug, topk_drug_ids)
        proc_tokens = _safe_index_select(h_proc, topk_proc_ids)

        if drug_tokens.size(1) != self.cfg.num_drug_tokens:
            raise ValueError(
                f"topk_drug_ids K={drug_tokens.size(1)} but cfg.num_drug_tokens={self.cfg.num_drug_tokens}"
            )
        if proc_tokens.size(1) != self.cfg.num_proc_tokens:
            raise ValueError(
                f"topk_proc_ids K={proc_tokens.size(1)} but cfg.num_proc_tokens={self.cfg.num_proc_tokens}"
            )

        patch = torch.cat([patient_token, drug_tokens, proc_tokens], dim=1)

        if self.type_embedding is not None:
            type_ids = torch.cat([
                torch.zeros((1,), dtype=torch.long, device=patch.device),
                torch.ones((self.cfg.num_drug_tokens,), dtype=torch.long, device=patch.device),
                torch.full((self.cfg.num_proc_tokens,), 2, dtype=torch.long, device=patch.device),
            ], dim=0)
            patch = patch + self.type_embedding(type_ids).unsqueeze(0)

        return patch

    def forward(
        self,
        x_patient: torch.Tensor,
        x_drug: torch.Tensor,
        x_proc: torch.Tensor,
        patient_drug_edge_index: torch.Tensor,
        patient_proc_edge_index: torch.Tensor,
        batch_patient_ids: torch.Tensor,
        topk_drug_ids: torch.Tensor,
        topk_proc_ids: torch.Tensor,
        return_embedding: bool = False,
    ):
        h_patient, h_drug, h_proc = self.encode_all_nodes(
            x_patient=x_patient,
            x_drug=x_drug,
            x_proc=x_proc,
            patient_drug_edge_index=patient_drug_edge_index,
            patient_proc_edge_index=patient_proc_edge_index,
        )

        patch = self.build_patch(
            h_patient=h_patient,
            h_drug=h_drug,
            h_proc=h_proc,
            batch_patient_ids=batch_patient_ids,
            topk_drug_ids=topk_drug_ids,
            topk_proc_ids=topk_proc_ids,
        )

        patch_out = self.patch_encoder(patch)
        patient_repr = patch_out[:, 0, :]
        logits = self.classifier(patient_repr)

        if return_embedding:
            return logits, patient_repr
        return logits


if __name__ == "__main__":
    torch.manual_seed(7)

    Np, Nd, Nr = 100, 50, 40
    Fp, Fd, Fr = 93, 131, 131
    C = 6
    Kd, Kr = 32, 31
    B = 8

    cfg = CARE4DDHGConvConfig(
        patient_in_dim=Fp,
        drug_in_dim=Fd,
        proc_in_dim=Fr,
        num_classes=C,
        hidden_dim=64,
        num_drug_tokens=Kd,
        num_proc_tokens=Kr,
        encoder_layers=3,
        dropout=0.1,
        use_hgconv=True,
    )

    model = CARE4DD_HGConv(cfg)

    x_p = torch.randn(Np, Fp)
    x_d = torch.randn(Nd, Fd)
    x_r = torch.randn(Nr, Fr)

    pd_edge = torch.stack([
        torch.randint(0, Np, (300,)),
        torch.randint(0, Nd, (300,)),
    ], dim=0)
    pr_edge = torch.stack([
        torch.randint(0, Np, (120,)),
        torch.randint(0, Nr, (120,)),
    ], dim=0)

    batch_pat = torch.randint(0, Np, (B,))
    topk_d = torch.randint(0, Nd, (B, Kd))
    topk_r = torch.randint(0, Nr, (B, Kr))

    logits, emb = model(
        x_patient=x_p,
        x_drug=x_d,
        x_proc=x_r,
        patient_drug_edge_index=pd_edge,
        patient_proc_edge_index=pr_edge,
        batch_patient_ids=batch_pat,
        topk_drug_ids=topk_d,
        topk_proc_ids=topk_r,
        return_embedding=True,
    )

    print("logits:", logits.shape)
    print("embedding:", emb.shape)
