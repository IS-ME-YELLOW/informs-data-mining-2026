"""Small dependency-free GAT for spatial residual learning."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _edge_softmax(scores, dst, n_nodes):
    """Softmax over incoming edges, separately for each attention head."""
    heads = scores.shape[1]
    flat_dst = dst[:, None] * heads + torch.arange(heads, device=scores.device)[None, :]
    flat_dst = flat_dst.reshape(-1)
    flat_scores = scores.reshape(-1)
    max_per = torch.full((n_nodes * heads,), -torch.inf, device=scores.device)
    max_per.scatter_reduce_(0, flat_dst, flat_scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(flat_scores - max_per[flat_dst])
    denom = torch.zeros(n_nodes * heads, device=scores.device)
    denom.index_add_(0, flat_dst, exp_scores)
    return (exp_scores / denom[flat_dst].clamp_min(1e-9)).reshape(-1, heads)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim * heads if concat else out_dim))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index):
        src, dst = edge_index
        z = self.linear(x).view(-1, self.heads, self.out_dim)
        score = (z[src] * self.att_src).sum(-1) + (z[dst] * self.att_dst).sum(-1)
        score = F.leaky_relu(score, negative_slope=0.2)
        alpha = _edge_softmax(score, dst, x.shape[0])
        alpha = self.dropout(alpha)
        out = torch.zeros_like(z)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * z[src])
        if self.concat:
            return out.reshape(x.shape[0], -1) + self.bias
        return out.mean(dim=1) + self.bias


class ResidualGAT(nn.Module):
    def __init__(self, in_dim, hidden=24, heads=4, dropout=0.12):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATLayer(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.skip = nn.Linear(in_dim, hidden, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index)) + self.skip(x)
        return self.head(h).squeeze(-1)


def block_edges(edge_index, n_nodes, n_snapshots, device):
    offsets = torch.arange(n_snapshots, device=device, dtype=torch.long) * n_nodes
    src = edge_index[0].to(device)[None, :] + offsets[:, None]
    dst = edge_index[1].to(device)[None, :] + offsets[:, None]
    return torch.stack((src.reshape(-1), dst.reshape(-1)), dim=0)


def fit_gat(features, residual, train_mask, edge_index, epochs=220, patience=35,
            hidden=24, heads=4, dropout=0.12, lr=0.003, weight_decay=2e-4,
            seed=42, device="cpu"):
    """Fit on a [time, county, feature] tensor and return model + predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(-1, features.shape[-1])
    y = torch.as_tensor(residual, dtype=torch.float32, device=device).reshape(-1)
    mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device).reshape(-1)
    if not bool(mask.any()):
        raise ValueError("GAT received no finite supervised county-time cells")
    edges = block_edges(torch.as_tensor(edge_index, dtype=torch.long), features.shape[1], features.shape[0], device)
    model = ResidualGAT(features.shape[-1], hidden=hidden, heads=heads, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_loss = float("inf")
    no_improve = 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, edges)
        # SmoothL1 limits the influence of a few exceptional outage counties.
        loss = F.smooth_l1_loss(pred[mask], y[mask], beta=0.01)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - 1e-7:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, edges).reshape(features.shape[0], features.shape[1]).cpu().numpy()
    return model, pred, {"epochs": epoch + 1, "best_loss": best_loss}


def predict_gat(model, features, edge_index, device="cpu"):
    """Run a fitted GAT on an arbitrary number of time snapshots."""
    device = torch.device(device)
    model = model.to(device).eval()
    x = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(-1, features.shape[-1])
    edges = block_edges(torch.as_tensor(edge_index, dtype=torch.long), features.shape[1], features.shape[0], device)
    with torch.no_grad():
        return model(x, edges).reshape(features.shape[0], features.shape[1]).cpu().numpy()
