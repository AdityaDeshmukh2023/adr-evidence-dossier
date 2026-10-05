"""Small symmetric classifiers. Import only in the optional Torch environment."""
from __future__ import annotations

import torch
from torch import nn

from . import MODEL_NAMES


def mean_neighbors(x, edge_index, edge_weight=None):
    """Sparse incoming-neighbor mean; gates retain original degree normalization."""
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    source, target = edge_index[0], edge_index[1]
    degree = torch.bincount(target, minlength=x.shape[0]).to(x.dtype).clamp_min(1)
    weights = degree[target].reciprocal()
    if edge_weight is not None:
        weights = weights * edge_weight
    adjacency = torch.sparse_coo_tensor(torch.stack((target, source)), weights,
                                        (x.shape[0], x.shape[0]), device=x.device, check_invariants=True).coalesce()
    return torch.sparse.mm(adjacency, x)


class MeanSage(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.root = nn.Linear(inputs, outputs)
        self.neighbor = nn.Linear(inputs, outputs, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        # No-bias linear transform commutes with mean; aggregate 128 values rather than 2048.
        return self.root(x) + mean_neighbors(self.neighbor(x), edge_index, edge_weight)


class PairClassifier(nn.Module):
    def __init__(self, model_name='fusion', role_dim=0, fingerprint_dim=2048,
                 hidden=128, embedding=64, dropout=.2):
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError('Unsupported classifier.')
        if model_name == 'role_fusion' and role_dim < 1:
            raise ValueError('Role-aware model requires actual biology features.')
        self.model_name, self.role_dim = model_name, role_dim
        self.dropout = nn.Dropout(dropout)
        if model_name != 'graphsage':
            self.molecular = nn.Sequential(nn.Linear(fingerprint_dim + (role_dim if model_name == 'role_fusion' else 0), hidden),
                                           nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, embedding), nn.ReLU())
        if model_name != 'fingerprint':
            self.sage1 = MeanSage(fingerprint_dim, hidden)
            self.sage2 = MeanSage(hidden, embedding)
        width = embedding * (2 if model_name in ('fusion', 'role_fusion') else 1)
        self.decoder = nn.Sequential(nn.Linear(width * 3, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 3))

    def encode(self, x, edge_index, roles=None, edge_weight=None):
        parts = []
        if self.model_name != 'graphsage':
            molecular_input = torch.cat((x, roles), dim=1) if self.model_name == 'role_fusion' else x
            parts.append(self.molecular(molecular_input))
        if self.model_name != 'fingerprint':
            h = self.dropout(torch.relu(self.sage1(x, edge_index, edge_weight)))
            parts.append(torch.relu(self.sage2(h, edge_index, edge_weight)))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def decode(self, embeddings, pairs):
        first, second = embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]
        symmetric = torch.cat((first + second, torch.abs(first - second), first * second), dim=1)
        return self.decoder(symmetric)

    def forward(self, x, edge_index, pairs, roles=None, edge_weight=None):
        return self.decode(self.encode(x, edge_index, roles, edge_weight), pairs)


def tensors(dataset, graph_pairs, device='cpu'):
    index = {row['ingredient_id']: i for i, row in enumerate(dataset['nodes'])}
    x = torch.zeros((len(index), dataset['fingerprint']['bits']), dtype=torch.float32, device=device)
    for position, row in enumerate(dataset['nodes']):
        x[position, row['fingerprint_bits']] = 1
    roles = torch.tensor([row['roles'] for row in dataset['nodes']], dtype=torch.float32, device=device)
    pairs = torch.tensor([[index[i] for i in row['ingredient_ids']] for row in dataset.get('pairs', [])],
                         dtype=torch.long, device=device).reshape(-1, 2)
    edges = [[index[first], index[second]] for first, second in graph_pairs]
    edges += [[second, first] for first, second in edges.copy()]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 2).t().contiguous()
    return index, x, roles, pairs, edge_index
