from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from core.graph_features import SPOILAGE_NODE_FEATURES

GNN_HIDDEN = 64
GNN_EMBED_DIM = 64


class SpoilageGNN(nn.Module):
    """GraphSAGE encoder (paper Alg 3): X=[N, 4] -> mean-pooled embedding z."""

    def __init__(
        self,
        in_dim: int = SPOILAGE_NODE_FEATURES,
        hidden: int = GNN_HIDDEN,
        embed_dim: int = GNN_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        return global_mean_pool(h, batch)


class SpoilagePretrainModel(nn.Module):
    """Encoder + linear head for supervised pretraining; only the encoder is kept."""

    def __init__(self, embed_dim: int = GNN_EMBED_DIM) -> None:
        super().__init__()
        self.encoder = SpoilageGNN(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.encoder(x, edge_index, batch)).squeeze(-1)
