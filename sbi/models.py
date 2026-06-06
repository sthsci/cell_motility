import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sinusoid_table(length: int, dim: int) -> torch.Tensor:
    """Return [1, length, dim] sinusoidal positional encodings."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
    )
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class TrajectoryTransformer(nn.Module):
    """
    Transformer encoder that maps a masked sequence of 2-D positions
    to a fixed-dimensional, ℓ2-normalised embedding.
    """

    def __init__(
        self,
        t_max: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        embed_dim: int = 32,
        num_classes: int = 32,
        input_dim: int = 2,
    ):
        super().__init__()
        self.t_max = t_max
        self.input_dim = input_dim
        self.coord_proj = nn.Linear(input_dim, d_model)

        # Not used for sinusoidal encoding
        # self.pos_emb = nn.Parameter(torch.zeros(1, t_max + 1, d_model))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.register_buffer(
            "pos_enc", _build_sinusoid_table(t_max + 1, d_model), persistent=False
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, embed_dim),
        )

        # self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.coord_proj.weight, std=0.02)
        if self.coord_proj.bias is not None:
            nn.init.zeros_(self.coord_proj.bias)

    # ---------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        assert T == self.t_max, (
            f"Expected time dimension {self.t_max}, got {T}. Pad or slice accordingly."
        )

        if mask is None:
            mask = x.new_zeros(B, T, dtype=torch.bool)

        x_proj = self.coord_proj(x)  # [B, T, d_model]
        x_proj = x_proj + self.pos_enc[:, 1 : T + 1, :]  # skip pos 0 (reserved for CLS)

        # prepend CLS token
        cls_tokens = (
            self.cls_token.expand(B, -1, -1) + self.pos_enc[:, :1, :]
        )  # Add positional encoding to CLS token
        x_proj = self.coord_proj(x) + self.pos_enc[:, 1 : T + 1, :]
        x_in = torch.cat([cls_tokens, x_proj], dim=1)

        # CLS is always valid, so prepend False
        key_padding = mask
        key_padding = F.pad(key_padding, (1, 0), value=False)  # [B, 1 + T]

        # transformer
        encoded = self.encoder(
            x_in, src_key_padding_mask=key_padding
        )  # [B, 1 + T, d_model]

        # take CLS output, project, normalise
        cls_out = encoded[:, 0]  # [B, d_model]
        emb = self.proj(cls_out)  # [B, embed_dim]
        emb = F.normalize(emb, p=2, dim=-1)  # unit-sphere

        logits = self.classifier(emb)

        return emb, logits
