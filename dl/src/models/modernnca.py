"""ModernNCA (Ye et al., ICLR 2025), re-implemented from the authors' code in LAMDA-TALENT. A house's price is
the softmax(-distance)-weighted mean of the prices of training houses, with distances measured in a learned
embedding space:

    phi(x) = [PLR(numerical columns) | binary + one-hot columns] -> Linear(dim) -> 0-2 MLP blocks
    y_hat(x) = sum_j softmax_j(-||phi(x) - phi(x_j)|| / T) * y_j,  T = 1, j over the candidate houses

PLR(lite) embeddings (Gorishniy et al., 2022) are applied to the continuous and ordinal columns; binary flags
and one-hot categoricals go in as they are, like TALENT's one-hot policy.

Candidates are always rows the model is trained on (inner_train during CV, the 85% fit part for the final
model), never validation or test rows. Training follows TALENT: each batch is compared with a random
sample_rate share of the other training rows plus the rest of the batch, each house's own label masked;
prediction uses every training row as a candidate."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from rtdl_num_embeddings import PeriodicEmbeddings
from torch import nn

from ..config import DEFAULT_N_TRIALS, SEED_MODEL
from ..utils import set_seed
from .base import BaseModel

NAME = "ModernNCA"
KIND = "onehot"
TEMPERATURE = 1.0
SMALL_DATA_ROWS = 2000  # below this many training rows use the small-data training config


def train_config(n_rows: int) -> dict:
    if n_rows < SMALL_DATA_ROWS:  # Housing, and small King County subsets in the learning curve
        return {"batch_size": 32, "max_epochs": 400, "patience": 30}
    # large batches keep the per-batch candidate encoding cheap; TALENT stops after ~20 epochs without progress
    return {"batch_size": 1024, "max_epochs": 200, "patience": 20}


def default_params(dataset: str) -> dict:
    """TALENT's default ModernNCA configuration (no MLP block, PLR-lite embeddings)."""
    return {"dim": 128, "n_blocks": 0, "sample_rate": 0.5, "n_frequencies": 77, "frequency_scale": 0.0443,
            "d_embedding": 34, "lr": 1e-2, "weight_decay": 2e-4, "scaler": "quantile"}


def n_trials(dataset: str) -> int:
    return DEFAULT_N_TRIALS[dataset]


def fixed_trials(dataset: str) -> list[dict]:
    """The default configuration, expressed with the Optuna parameter names, is always evaluated."""
    return [{**default_params(dataset), "use_weight_decay": True}]


def suggest_params(trial, dataset: str) -> dict:
    """TALENT's ModernNCA search space; dimensions narrowed for the 545-row Housing dataset. Block width and
    dropout only exist when there is at least one MLP block."""
    d_max = 512 if dataset == "housing" else 1024
    n_blocks = trial.suggest_int("n_blocks", 0, 2)
    p = {"dim": trial.suggest_int("dim", 64, d_max), "n_blocks": n_blocks}
    if n_blocks:
        p["d_block"] = trial.suggest_int("d_block", 64, d_max)
        p["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    use_wd = trial.suggest_categorical("use_weight_decay", [False, True])  # {0, LogUniform[1e-6, 1e-3]}
    return {
        **p,
        "sample_rate": trial.suggest_float("sample_rate", 0.05, 0.6),
        "n_frequencies": trial.suggest_int("n_frequencies", 16, 96),
        "frequency_scale": trial.suggest_float("frequency_scale", 0.005, 10.0, log=True),
        "d_embedding": trial.suggest_int("d_embedding", 16, 64),
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) if use_wd else 0.0,
        "scaler": trial.suggest_categorical("scaler", ["quantile", "standard"]),
    }


def adapt_to_size(params: dict, n_rows: int) -> dict:
    return params  # sample_rate >= 0.05 still leaves candidates at 436 rows; the batch size follows train_config


class _Block(nn.Module):
    """TALENT's MLP block (no residual connection)."""

    def __init__(self, dim: int, d_block: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, d_block), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(d_block, dim))

    def forward(self, x):
        return self.block(x)


class _Encoder(nn.Module):
    def __init__(self, params: dict, n_features: int, n_plr: int):
        super().__init__()
        p, self.n_plr = params, n_plr
        self.plr = (PeriodicEmbeddings(n_plr, p["d_embedding"], n_frequencies=p["n_frequencies"],
                                       frequency_init_scale=p["frequency_scale"], lite=True) if n_plr else None)
        d_in = n_plr * p["d_embedding"] + n_features - n_plr if n_plr else n_features
        self.linear = nn.Linear(d_in, p["dim"])
        self.blocks = (nn.Sequential(*[_Block(p["dim"], p["d_block"], p["dropout"]) for _ in range(p["n_blocks"])],
                                     nn.BatchNorm1d(p["dim"])) if p["n_blocks"] else nn.Identity())

    def forward(self, x):
        if self.plr is not None:
            x = torch.cat([self.plr(x[:, :self.n_plr]).flatten(1), x[:, self.n_plr:]], dim=1)
        return self.blocks(self.linear(x))


class ModernNCAModel(BaseModel):
    kind = KIND

    def __init__(self, params: dict, n_features: int, n_plr: int, device: str = "cpu"):
        self.params, self.n_features, self.n_plr, self.device = params, n_features, n_plr, device
        self.history_ = {"train_loss": [], "val_rmse": []}

    def _network(self) -> _Encoder:
        set_seed(SEED_MODEL)
        return _Encoder(self.params, self.n_features, self.n_plr).to(self.device)

    @torch.no_grad()
    def _encode(self, net, x, batch_size: int = 8192) -> torch.Tensor:
        net.eval()
        return torch.cat([net(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])

    @torch.no_grad()
    def _weights(self, net, x, batch_size: int = 1024):
        """Softmax weights of every candidate (training row) for each row of x, batch by batch."""
        cand = self._encode(net, self.cand_x_)
        z = self._encode(net, x)
        for i in range(0, len(z), batch_size):
            yield F.softmax(-torch.cdist(z[i:i + batch_size], cand) / TEMPERATURE, dim=-1)

    def _predict(self, net, x) -> torch.Tensor:
        return torch.cat([w @ self.cand_y_ for w in self._weights(net, x)])

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, n_epochs=None):
        p, cfg = self.params, train_config(len(y_tr))
        net = self._network()
        optimizer = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        self.cand_x_ = torch.as_tensor(X_tr, dtype=torch.float32, device=self.device)
        self.cand_y_ = torch.as_tensor(y_tr, dtype=torch.float32, device=self.device)
        x, y = self.cand_x_, self.cand_y_
        has_val = X_val is not None
        if has_val:
            xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        max_epochs = cfg["max_epochs"] if has_val else int(n_epochs)

        generator = torch.Generator().manual_seed(SEED_MODEL)
        best_rmse, best_state, best_epoch = math.inf, None, 0
        n, bs = len(y), cfg["batch_size"]
        for epoch in range(1, max_epochs + 1):
            net.train()
            perm = torch.randperm(n, generator=generator)
            total = 0.0
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                if len(b) < 2:  # BatchNorm of the MLP blocks cannot train on one row
                    continue
                others = torch.ones(n, dtype=torch.bool)
                others[b] = False
                others = others.nonzero().squeeze(1)
                n_sample = min(len(others), max(2, int(len(others) * p["sample_rate"])))
                c = others[torch.randperm(len(others), generator=generator)[:n_sample]]
                b, c = b.to(self.device), c.to(self.device)
                zb = net(x[b])
                cand = torch.cat([zb, net(x[c])]) if len(c) else zb
                dist = torch.cdist(zb, cand) / TEMPERATURE
                dist.fill_diagonal_(math.inf)  # a house never retrieves its own label
                pred = F.softmax(-dist, dim=-1) @ torch.cat([y[b], y[c]])
                loss = F.mse_loss(pred, y[b])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(b)
            train_loss = total / n
            self.history_["train_loss"].append(train_loss)
            if not math.isfinite(train_loss):  # diverged (lr too high): keep the best weights so far
                break
            if has_val:
                val_rmse = torch.sqrt(F.mse_loss(self._predict(net, xv), yv)).item()
                self.history_["val_rmse"].append(val_rmse)
                if val_rmse < best_rmse:
                    best_rmse, best_epoch = val_rmse, epoch
                    best_state = copy.deepcopy(net.state_dict())
                elif epoch - best_epoch >= cfg["patience"]:
                    break
        if has_val and best_state is not None:
            net.load_state_dict(best_state)
        self.best_epoch_ = best_epoch if has_val else max_epochs
        self.net_ = net
        return self

    def predict(self, X) -> np.ndarray:
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict(self.net_, x).cpu().numpy()

    def neighbor_weights(self, X) -> np.ndarray:
        """(rows of X, training rows) softmax weights, for the explanation by nearest training houses."""
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return torch.cat(list(self._weights(self.net_, x))).cpu().numpy()

    def save(self, path) -> None:
        # the candidate set is part of the model: predictions are weighted means of its labels
        torch.save({"state_dict": self.net_.state_dict(), "params": self.params, "n_features": self.n_features,
                    "n_plr": self.n_plr, "cand_x": self.cand_x_.cpu(), "cand_y": self.cand_y_.cpu()}, f"{path}.pt")


def build(params: dict, pre, dataset: str, device: str) -> ModernNCAModel:
    # PLR for the continuous and ordinal columns, which come first in the one-hot matrix
    return ModernNCAModel(params, n_features=len(pre.feature_names), n_plr=len(pre.spec.cont) + len(pre.spec.ord),
                          device=device)
