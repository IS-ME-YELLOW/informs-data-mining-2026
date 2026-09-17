"""Small edge-aware GAT with deterministic best-checkpoint semantics."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    # Determinism is part of the diagnostic contract.  If the selected device
    # has no deterministic implementation for an operator, let Torch raise
    # instead of silently weakening the claim.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _require_finite(name, value) -> None:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"non-finite {name}")


def _edge_softmax(scores, dst, n_nodes):
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
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.1, edge_dim=4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.edge_linear = nn.Linear(edge_dim, heads, bias=False) if edge_dim else None
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim * heads if concat else out_dim))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, edge_attr=None):
        src, dst = edge_index
        z = self.linear(x).view(-1, self.heads, self.out_dim)
        score = (z[src] * self.att_src).sum(-1) + (z[dst] * self.att_dst).sum(-1)
        if self.edge_linear is not None:
            if edge_attr is None:
                raise ValueError("edge_attr is required by this GAT layer")
            score = score + self.edge_linear(edge_attr)
        score = F.leaky_relu(score, negative_slope=0.2)
        alpha = self.dropout(_edge_softmax(score, dst, x.shape[0]))
        out = torch.zeros_like(z)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * z[src])
        if self.concat:
            return out.reshape(x.shape[0], -1) + self.bias
        return out.mean(dim=1) + self.bias


class ResidualGAT(nn.Module):
    def __init__(self, in_dim, hidden=24, heads=4, dropout=0.12, edge_dim=4):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden, heads=heads, concat=True, dropout=dropout, edge_dim=edge_dim)
        self.gat2 = GATLayer(hidden * heads, hidden, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.skip = nn.Linear(in_dim, hidden, bias=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr=None):
        h = F.elu(self.gat1(x, edge_index, edge_attr))
        h = F.elu(self.gat2(h, edge_index, edge_attr)) + self.skip(x)
        return self.head(h).squeeze(-1)


def block_edges(edge_index, n_nodes, n_snapshots, device):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.shape[1] == 0 or bool((edge_index < 0).any()) or bool((edge_index >= n_nodes).any()):
        raise ValueError("edge_index contains an invalid node id")
    offsets = torch.arange(n_snapshots, device=device, dtype=torch.long) * n_nodes
    src = edge_index[0][None, :] + offsets[:, None]
    dst = edge_index[1][None, :] + offsets[:, None]
    return torch.stack((src.reshape(-1), dst.reshape(-1)), dim=0)


def block_edge_attributes(edge_attr, n_snapshots, device):
    if edge_attr is None:
        return None
    value = torch.as_tensor(edge_attr, dtype=torch.float32, device=device)
    if value.ndim != 2 or value.shape[0] == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("edge_attr must be a finite [E, D] array")
    return value.repeat(n_snapshots, 1)


def _loss(pred, y, mask, loss_mode):
    errors = pred[mask] - y[mask]
    if loss_mode == "mse":
        return torch.mean(errors ** 2)
    if loss_mode == "weighted_mse":
        weights = 1.0 + 4.0 * torch.clamp(torch.abs(y[mask]) / 2.5, 0.0, 1.0)
        return torch.mean(weights * errors ** 2)
    if loss_mode == "huber":
        return F.smooth_l1_loss(pred[mask], y[mask])
    raise ValueError(f"unknown loss_mode={loss_mode!r}")


def fit_gat(
    features,
    residual,
    train_mask,
    edge_index,
    epochs=220,
    patience=35,
    hidden=24,
    heads=4,
    dropout=0.12,
    lr=0.003,
    weight_decay=2e-4,
    seed=42,
    device="cpu",
    edge_attr=None,
    loss_mode="mse",
):
    """Fit on [time, county, feature] and return deterministic predictions.

    Early stopping is based only on the supplied supervision mask. The loss is
    recomputed after every optimizer step in eval/no_grad mode, so the saved
    state, best_loss and best_epoch refer to the same deterministic model.
    """

    if int(epochs) < 1 or int(patience) < 1:
        raise ValueError("epochs and patience must both be >= 1")
    if loss_mode not in {"mse", "weighted_mse", "huber"}:
        raise ValueError(f"unknown loss_mode={loss_mode!r}")
    features = np.asarray(features, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    train_mask = np.asarray(train_mask, dtype=bool)
    if features.ndim != 3 or residual.shape != features.shape[:2] or train_mask.shape != features.shape[:2]:
        raise ValueError("features, residual and train_mask have incompatible shapes")
    if not np.isfinite(features).all() or not np.isfinite(residual[train_mask]).all():
        raise FloatingPointError("GAT received non-finite features or supervised residual")
    if not train_mask.any():
        raise ValueError("GAT received no finite supervised county-time cells")
    seed_everything(seed)
    device = torch.device(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(-1, features.shape[-1])
    y = torch.as_tensor(residual, dtype=torch.float32, device=device).reshape(-1)
    mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device).reshape(-1)
    edges = block_edges(edge_index, features.shape[1], features.shape[0], device)
    block_attr = block_edge_attributes(edge_attr, features.shape[0], device)
    if block_attr is not None and block_attr.shape[0] != edges.shape[1]:
        raise ValueError("edge_attr row count does not match edge_index")
    model = ResidualGAT(
        features.shape[-1], hidden=hidden, heads=heads, dropout=dropout,
        edge_dim=block_attr.shape[-1] if block_attr is not None else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, edges, block_attr)
        _require_finite("training prediction", pred)
        loss = _loss(pred, y, mask, loss_mode)
        _require_finite("training loss", loss)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                _require_finite("gradient", parameter.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        for parameter in model.parameters():
            _require_finite("parameter", parameter)

        model.eval()
        with torch.no_grad():
            check_pred = model(x, edges, block_attr)
            check_loss = _loss(check_pred, y, mask, "mse")
        _require_finite("deterministic training prediction", check_pred)
        _require_finite("deterministic training loss", check_loss)
        value = float(check_loss.detach().cpu())
        if value < best_loss - 1e-7:
            best_loss = value
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"[GAT] epoch={epoch + 1}/{int(epochs)} "
            f"train_loss={float(loss.detach().cpu()):.8f} "
            f"check_loss={value:.8f} best_loss={best_loss:.8f} "
            f"no_improve={no_improve}/{int(patience)}",
            flush=True,
        )
        if no_improve >= int(patience):
            break

    if best_state is None:
        raise RuntimeError("GAT did not produce a finite best checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = model(x, edges, block_attr).reshape(features.shape[0], features.shape[1])
    _require_finite("final prediction", final_pred)
    return model, final_pred.cpu().numpy(), {
        "epochs": epoch + 1,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "loss_mode": loss_mode,
        "seed": int(seed),
    }


def predict_gat(model, features, edge_index, device="cpu", edge_attr=None):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or not np.isfinite(features).all():
        raise FloatingPointError("GAT inference received non-finite features")
    device = torch.device(device)
    model = model.to(device).eval()
    x = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(-1, features.shape[-1])
    edges = block_edges(edge_index, features.shape[1], features.shape[0], device)
    block_attr = block_edge_attributes(edge_attr, features.shape[0], device)
    if block_attr is not None and block_attr.shape[0] != edges.shape[1]:
        raise ValueError("edge_attr row count does not match edge_index")
    with torch.no_grad():
        prediction = model(x, edges, block_attr).reshape(features.shape[0], features.shape[1])
    _require_finite("GAT inference prediction", prediction)
    return prediction.cpu().numpy()
