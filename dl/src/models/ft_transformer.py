"""FT-Transformer (Gorishniy et al., NeurIPS 2021) via the authors' rtdl_revisiting_models package,
with our own training loop (AdamW, MSE, early stopping on inner_val)."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from rtdl_revisiting_models import FTTransformer

from ..config import SEED_MODEL
from ..utils import set_seed
from .base import BaseModel

NAME = "FT-Transformer"
KIND = "numcat"
N_HEADS = 8
SMALL_DATA_ROWS = 2000  # below this many training rows use the small-data training config


def train_config(n_rows: int) -> dict:
    if n_rows < SMALL_DATA_ROWS:  # Housing, and small King County subsets in the learning curve
        return {"batch_size": 32, "max_epochs": 400, "patience": 30}
    return {"batch_size": 256, "max_epochs": 200, "patience": 16}


def default_params(dataset: str) -> dict:
    """Default FT-Transformer of the paper (3 blocks, d=192)."""
    return {"n_blocks": 3, "d_block": 192, "attention_dropout": 0.2, "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": 0.1, "residual_dropout": 0.0, "lr": 1e-4, "weight_decay": 1e-5, "scaler": "quantile"}


def default_trial_params(dataset: str) -> dict:
    """default_params() expressed with the Optuna parameter names (enqueued as trial 0)."""
    p = default_params(dataset)
    del p["residual_dropout"]
    return {**p, "use_residual_dropout": False}


def suggest_params(trial, dataset: str) -> dict:
    """Search space of the paper's appendix, narrowed for the 545-row Housing dataset."""
    if dataset == "housing":
        n_blocks = trial.suggest_int("n_blocks", 1, 3)
        d_block = trial.suggest_int("d_block", 32, 256, step=32)
    else:
        n_blocks = trial.suggest_int("n_blocks", 1, 4)
        d_block = trial.suggest_int("d_block", 64, 512, step=64)
    use_residual = trial.suggest_categorical("use_residual_dropout", [False, True])
    return {
        "n_blocks": n_blocks,
        "d_block": d_block,
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
        "ffn_d_hidden_multiplier": trial.suggest_float("ffn_d_hidden_multiplier", 2 / 3, 8 / 3),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
        "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2) if use_residual else 0.0,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "scaler": trial.suggest_categorical("scaler", ["quantile", "standard"]),
    }


def adapt_to_size(params: dict, n_rows: int) -> dict:
    return params  # the training config already follows the number of rows


class FTTransformerModel(BaseModel):
    kind = KIND

    def __init__(self, params: dict, n_cont: int, cat_cardinalities: list[int], device: str = "cpu"):
        self.params, self.n_cont, self.cat_cardinalities, self.device = params, n_cont, cat_cardinalities, device
        self.history_ = {"train_loss": [], "val_rmse": []}

    def _network(self) -> FTTransformer:
        p = self.params
        set_seed(SEED_MODEL)
        return FTTransformer(
            n_cont_features=self.n_cont,
            cat_cardinalities=self.cat_cardinalities,
            d_out=1,
            n_blocks=p["n_blocks"],
            d_block=p["d_block"],
            attention_n_heads=N_HEADS,
            attention_dropout=p["attention_dropout"],
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=p["ffn_d_hidden_multiplier"],
            ffn_dropout=p["ffn_dropout"],
            residual_dropout=p["residual_dropout"],
        ).to(self.device)

    def _tensors(self, X):
        x_num, x_cat = X
        xn = torch.as_tensor(x_num, dtype=torch.float32, device=self.device)
        xc = None if x_cat is None else torch.as_tensor(x_cat, dtype=torch.long, device=self.device)
        return xn, xc

    @torch.no_grad()
    def _predict(self, net, xn, xc, batch_size: int = 4096) -> torch.Tensor:
        net.eval()
        out = [net(xn[i:i + batch_size], None if xc is None else xc[i:i + batch_size]).squeeze(-1)
               for i in range(0, len(xn), batch_size)]
        return torch.cat(out)

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, n_epochs=None):
        p, cfg = self.params, train_config(len(y_tr))
        net = self._network()
        optimizer = torch.optim.AdamW(net.make_parameter_groups(), lr=p["lr"], weight_decay=p["weight_decay"])
        xn, xc = self._tensors(X_tr)
        y = torch.as_tensor(y_tr, dtype=torch.float32, device=self.device)
        has_val = X_val is not None
        if has_val:
            vn, vc = self._tensors(X_val)
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
                loss = F.mse_loss(net(xn[b], None if xc is None else xc[b]).squeeze(-1), y[b])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(b)
            train_loss = total / n
            self.history_["train_loss"].append(train_loss)
            if not math.isfinite(train_loss):  # diverged (lr too high): keep the best weights so far
                break
            if has_val:
                val_rmse = torch.sqrt(F.mse_loss(self._predict(net, vn, vc), yv)).item()
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
        xn, xc = self._tensors(X)
        return self._predict(self.net_, xn, xc).cpu().numpy()

    def save(self, path) -> None:
        torch.save(self.net_.state_dict(), f"{path}.pt")


def build(params: dict, pre, dataset: str, device: str) -> FTTransformerModel:
    return FTTransformerModel(params, n_cont=pre.n_num, cat_cardinalities=pre.cat_cardinalities_, device=device)
