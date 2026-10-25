"""TabNet (Arik & Pfister, AAAI 2021) via pytorch-tabnet, wrapped in the common interface."""
import contextlib
import io
import warnings

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

from ..config import SEED_MODEL
from ..utils import set_seed
from .base import BaseModel

NAME = "TabNet"
KIND = "tabnet"
MAX_EPOCHS = 300
PATIENCE = 30
SMALL_DATA_ROWS = 2000
LOW_CARDINALITY = 10  # categorical columns with few levels get a 2-dim embedding


def default_params(dataset: str) -> dict:
    p = {"n_d": 16, "n_steps": 4, "gamma": 1.3, "n_independent": 2, "n_shared": 2, "lambda_sparse": 1e-4,
         "momentum": 0.02, "lr": 0.02, "mask_type": "sparsemax", "scaler": "quantile"}
    if dataset == "housing":  # library default batch 1024 is larger than a Housing fold
        p.update(batch_size=64, virtual_batch_size=16)
    else:
        p.update(batch_size=256, virtual_batch_size=64, cat_emb_dim=8)
    return p


def default_trial_params(dataset: str) -> dict:
    return default_params(dataset)


def suggest_params(trial, dataset: str) -> dict:
    small = dataset == "housing"
    p = {
        "n_d": trial.suggest_categorical("n_d", [8, 16, 24] if small else [8, 16, 32, 64]),  # n_a = n_d
        "n_steps": trial.suggest_int("n_steps", 3, 5 if small else 8),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "n_independent": trial.suggest_int("n_independent", 1, 2 if small else 3),
        "n_shared": trial.suggest_int("n_shared", 1, 2 if small else 3),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
        "momentum": trial.suggest_float("momentum", 0.01, 0.4),
        "lr": trial.suggest_categorical("lr", [0.005, 0.01, 0.02] if small else [0.005, 0.01, 0.02, 0.025]),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128] if small else [256, 512, 1024]),
        "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [16, 32] if small else [64, 128, 256]),
        "scaler": trial.suggest_categorical("scaler", ["quantile", "standard"]),
    }
    if not small:
        p["cat_emb_dim"] = trial.suggest_categorical("cat_emb_dim", [4, 8])
    return p


def adapt_to_size(params: dict, n_rows: int) -> dict:
    """Learning curve: King County batch sizes (>=256) leave only 1-2 steps per epoch on tiny subsets."""
    if n_rows < SMALL_DATA_ROWS:
        return {**params, "batch_size": 64, "virtual_batch_size": 16}
    return params


class TabNetModel(BaseModel):
    kind = KIND

    def __init__(self, params: dict, cat_idxs: list[int], cat_dims: list[int], device: str = "cpu"):
        self.params, self.cat_idxs, self.cat_dims, self.device = params, cat_idxs, cat_dims, device
        self.history_ = {"train_loss": [], "val_rmse": []}

    def _regressor(self) -> TabNetRegressor:
        p = self.params
        emb = [2 if d <= LOW_CARDINALITY else p.get("cat_emb_dim", 8) for d in self.cat_dims]
        set_seed(SEED_MODEL)
        return TabNetRegressor(
            n_d=p["n_d"], n_a=p["n_d"], n_steps=p["n_steps"], gamma=p["gamma"],
            n_independent=p["n_independent"], n_shared=p["n_shared"],
            lambda_sparse=p["lambda_sparse"], momentum=p["momentum"], mask_type=p["mask_type"],
            cat_idxs=self.cat_idxs, cat_dims=self.cat_dims, cat_emb_dim=emb,
            optimizer_fn=torch.optim.Adam, optimizer_params={"lr": p["lr"]},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={"step_size": 50, "gamma": 0.9},
            seed=SEED_MODEL, verbose=0, device_name=self.device,
        )

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, n_epochs=None):
        p = self.params
        bs = p["batch_size"]
        common = {"batch_size": bs, "virtual_batch_size": min(p["virtual_batch_size"], bs),
                  "drop_last": len(y_tr) % bs == 1,  # a batch of one row breaks BatchNorm
                  "num_workers": 0}
        self.reg_ = self._regressor()
        # pytorch-tabnet prints and warns on every fit even with verbose=0
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            if X_val is not None:
                self.reg_.fit(X_tr, y_tr.reshape(-1, 1), eval_set=[(X_val, y_val.reshape(-1, 1))], eval_name=["val"],
                              eval_metric=["rmse"], max_epochs=MAX_EPOCHS, patience=PATIENCE, **common)
                self.best_epoch_ = int(self.reg_.best_epoch) + 1
            else:
                self.reg_.fit(X_tr, y_tr.reshape(-1, 1), max_epochs=int(n_epochs), **common)
                self.best_epoch_ = int(n_epochs)
        hist = self.reg_.history.history
        self.history_ = {"train_loss": [float(v) for v in hist["loss"]],
                         "val_rmse": [float(v) for v in hist.get("val_rmse", [])]}
        return self

    def predict(self, X) -> np.ndarray:
        return self.reg_.predict(X).ravel()

    def explain(self, X) -> tuple[np.ndarray, dict]:
        """Per-row aggregated importance (n, n_features) and the mask of every decision step."""
        return self.reg_.explain(X)

    def save(self, path) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.reg_.save_model(str(path))


def build(params: dict, pre, dataset: str, device: str) -> TabNetModel:
    return TabNetModel(params, cat_idxs=pre.cat_idxs, cat_dims=pre.cat_cardinalities_, device=device)
