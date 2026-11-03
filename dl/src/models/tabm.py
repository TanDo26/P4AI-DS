"""TabM (Gorishniy et al., ICLR 2025) via the authors' `tabm` package: one network holding an ensemble of
k = 32 MLPs that share most of their weights. It takes exactly the MLP's input (one-hot, KIND = "onehot", no
numerical embeddings), so MLP vs TabM isolates the effect of the ensemble. Training follows the package's
example: AdamW, MSE on each of the k predictions separately, gradient clipping 1.0, early stopping on the mean
of the k predictions."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from tabm import TabM

from ..config import DEFAULT_N_TRIALS, SEED_MODEL
from ..utils import set_seed
from .base import BaseModel

NAME = "TabM"
KIND = "onehot"
K = 32  # ensemble size, fixed (not tuned) as recommended by the authors
GRAD_CLIP = 1.0
SMALL_DATA_ROWS = 2000  # below this many training rows use the small-data training config


def train_config(n_rows: int) -> dict:
    if n_rows < SMALL_DATA_ROWS:  # Housing, and small King County subsets in the learning curve
        return {"batch_size": 32, "max_epochs": 400, "patience": 30}
    return {"batch_size": 256, "max_epochs": 200, "patience": 16}  # batch 256, patience 16: the package example


def default_params(dataset: str) -> dict:
    """TabM.make defaults (3 blocks of 512, dropout 0.1) and the optimizer of the package example."""
    return {"n_blocks": 3, "d_block": 512, "dropout": 0.1, "lr": 2e-3, "weight_decay": 3e-4, "scaler": "quantile"}


def n_trials(dataset: str) -> int:
    return DEFAULT_N_TRIALS[dataset]


def fixed_trials(dataset: str) -> list[dict]:
    """The default configuration, expressed with the Optuna parameter names, is always evaluated."""
    p = default_params(dataset)
    return [{**p, "use_weight_decay": True}]


def suggest_params(trial, dataset: str) -> dict:
    """Search space of the TabM README (k fixed at 32), plus dropout; narrowed for the 545-row Housing dataset."""
    if dataset == "housing":
        n_blocks = trial.suggest_int("n_blocks", 1, 3)
        d_block = trial.suggest_int("d_block", 64, 512, step=16)
    else:
        n_blocks = trial.suggest_int("n_blocks", 1, 5)
        d_block = trial.suggest_int("d_block", 64, 1024, step=16)
    use_wd = trial.suggest_categorical("use_weight_decay", [False, True])  # {0, LogUniform[1e-4, 1e-1]}
    return {
        "n_blocks": n_blocks,
        "d_block": d_block,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True) if use_wd else 0.0,
        "scaler": trial.suggest_categorical("scaler", ["quantile", "standard"]),
    }


def adapt_to_size(params: dict, n_rows: int) -> dict:
    return params  # the training config already follows the number of rows


class TabMModel(BaseModel):
    kind = KIND

    def __init__(self, params: dict, n_features: int, device: str = "cpu"):
        self.params, self.n_features, self.device = params, n_features, device
        self.history_ = {"train_loss": [], "val_rmse": []}

    def _network(self) -> TabM:
        p = self.params
        set_seed(SEED_MODEL)
        return TabM.make(n_num_features=self.n_features, d_out=1, k=K, n_blocks=p["n_blocks"],
                         d_block=p["d_block"], dropout=p["dropout"]).to(self.device)

    @torch.no_grad()
    def _predict(self, net, x, batch_size: int = 4096) -> torch.Tensor:
        """Mean of the k predictions (on the standardized log-price scale the model is trained on)."""
        net.eval()
        return torch.cat([net(x[i:i + batch_size]).squeeze(-1).mean(1) for i in range(0, len(x), batch_size)])

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, n_epochs=None):
        p, cfg = self.params, train_config(len(y_tr))
        net = self._network()
        optimizer = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        x = torch.as_tensor(X_tr, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_tr, dtype=torch.float32, device=self.device)
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
            perm = torch.randperm(n, generator=generator).to(self.device)
            total = 0.0
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                # every member is trained on the batch: the mean loss, not the loss of the mean prediction
                pred = net(x[b]).squeeze(-1)  # (batch, k)
                loss = F.mse_loss(pred.flatten(), y[b].repeat_interleave(K))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
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

    def save(self, path) -> None:
        torch.save({"state_dict": self.net_.state_dict(), "params": self.params, "n_features": self.n_features},
                   f"{path}.pt")


def build(params: dict, pre, dataset: str, device: str) -> TabMModel:
    return TabMModel(params, n_features=len(pre.feature_names), device=device)
