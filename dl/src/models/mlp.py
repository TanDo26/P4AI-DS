"""Feed-forward MLP ported from the team's original MLP code (dl_mlp_project): the same architecture and
training (ReLU + dropout, AdamW, MSE, ReduceLROnPlateau, gradient clipping, early stopping with patience 22),
on the shared features with one-hot categoricals (KIND = "onehot"). Tuned with Optuna like the other models;
the original code's 5 configurations run first, as fixed trials."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import DEFAULT_N_TRIALS, SEED_MODEL
from ..utils import set_seed
from .base import BaseModel

NAME = "MLP"
KIND = "onehot"
MAX_EPOCHS = 200
PATIENCE = 22
MIN_DELTA = 1e-6  # an epoch only counts as an improvement if inner_val MSE drops by more than this
GRAD_CLIP = 5.0

# the original code's grid (hidden layers, dropout, AdamW learning rate / weight decay, batch size),
# now the first 5 trials of the Optuna study
CONFIGS = {
    "mlp_64_32_d0_lr1e-3_wd1e-4_b32":
        {"hidden_dims": [64, 32], "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 32},
    "mlp_128_64_d0.1_lr1e-3_wd1e-4_b64":
        {"hidden_dims": [128, 64], "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64},
    "mlp_128_64_32_d0.2_lr5e-4_wd1e-3_b128":
        {"hidden_dims": [128, 64, 32], "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-3, "batch_size": 128},
    "mlp_256_128_d0.1_lr5e-4_wd1e-4_b128":
        {"hidden_dims": [256, 128], "dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 128},
    "mlp_64_32_d0.2_lr1e-3_wd1e-2_b64":
        {"hidden_dims": [64, 32], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64},
}


def _trial_params(config: dict) -> dict:
    """An original configuration expressed with the Optuna parameter names (StandardScaler, as in the original)."""
    dims = config["hidden_dims"]
    p = {"n_layers": len(dims), "d_first": dims[0]}
    if len(dims) >= 3:
        p["d_middle"] = dims[1]
    if len(dims) >= 2:
        p["d_last"] = dims[-1]
    return {**p, **{k: config[k] for k in ("dropout", "lr", "weight_decay", "batch_size")}, "scaler": "standard"}


def n_trials(dataset: str) -> int:
    return DEFAULT_N_TRIALS[dataset]


def default_params(dataset: str) -> dict:
    return {**next(iter(CONFIGS.values())), "scaler": "standard"}


def fixed_trials(dataset: str) -> list[dict]:
    """The 5 original configurations always run first, so the study contains the original grid."""
    return [_trial_params(c) for c in CONFIGS.values()]


def suggest_params(trial, dataset: str) -> dict:
    """MLP search space of Gorishniy et al. (2021): widths of the first, middle (shared) and last layers.
    Narrowed to at most 4 layers; weight decay widened to 1e-2 and batch sizes 32-256 so that the 5 original
    configurations lie inside the space."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    d_first = trial.suggest_int("d_first", 16, 512, log=True)
    d_middle = trial.suggest_int("d_middle", 16, 512, log=True) if n_layers >= 3 else None
    d_last = trial.suggest_int("d_last", 16, 512, log=True) if n_layers >= 2 else None
    return {
        "hidden_dims": [d_first] + [d_middle] * (n_layers - 2) + ([d_last] if n_layers >= 2 else []),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "scaler": trial.suggest_categorical("scaler", ["standard", "quantile"]),
    }


def adapt_to_size(params: dict, n_rows: int) -> dict:
    return params  # batch sizes up to 256 still give 2+ batches on the smallest learning-curve subset (436 rows)


class MLPModel(BaseModel):
    kind = KIND

    def __init__(self, params: dict, n_features: int, device: str = "cpu"):
        self.params, self.n_features, self.device = params, n_features, device
        self.history_ = {"train_loss": [], "val_rmse": []}

    def _network(self) -> nn.Sequential:
        p = self.params
        set_seed(SEED_MODEL)
        layers, width = [], self.n_features
        for hidden in p["hidden_dims"]:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            if p["dropout"] > 0:
                layers.append(nn.Dropout(p["dropout"]))
            width = hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers).to(self.device)

    @torch.no_grad()
    def _predict(self, net, x, batch_size: int = 8192) -> torch.Tensor:
        net.eval()
        return torch.cat([net(x[i:i + batch_size]).squeeze(-1) for i in range(0, len(x), batch_size)])

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, n_epochs=None):
        p = self.params
        net = self._network()
        optimizer = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7,
                                                               min_lr=1e-5)
        x = torch.as_tensor(X_tr, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_tr, dtype=torch.float32, device=self.device)
        has_val = X_val is not None
        if has_val:
            xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            yv = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        max_epochs = MAX_EPOCHS if has_val else int(n_epochs)

        # batches reshuffled every epoch by a generator seeded 36, like the original DataLoader(shuffle=True)
        generator = torch.Generator().manual_seed(SEED_MODEL)
        best_mse, best_state, best_epoch, wait = math.inf, copy.deepcopy(net.state_dict()), 1, 0
        n, bs = len(y), p["batch_size"]
        for epoch in range(1, max_epochs + 1):
            net.train()
            perm = torch.randperm(n, generator=generator).to(self.device)
            losses = []
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                loss = F.mse_loss(net(x[b]).squeeze(-1), y[b])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                losses.append(loss.item())
            self.history_["train_loss"].append(float(np.mean(losses)))
            if not has_val:
                continue
            val_mse = F.mse_loss(self._predict(net, xv), yv).item()
            self.history_["val_rmse"].append(math.sqrt(val_mse))
            scheduler.step(val_mse)
            if val_mse < best_mse - MIN_DELTA:
                best_mse, best_epoch, wait = val_mse, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if has_val:
            net.load_state_dict(best_state)
        self.best_epoch_ = best_epoch if has_val else max_epochs
        self.net_ = net
        return self

    def predict(self, X) -> np.ndarray:
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict(self.net_, x).cpu().numpy()

    def save(self, path) -> None:
        torch.save({"state_dict": self.net_.state_dict(), "params": self.params, "n_features": self.n_features},
                   f"{path}.pt")


def build(params: dict, pre, dataset: str, device: str) -> MLPModel:
    return MLPModel(params, n_features=len(pre.feature_names), device=device)
