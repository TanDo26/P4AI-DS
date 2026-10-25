"""Metrics on the original price scale, plus bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import N_BOOTSTRAP, SEED_DATA


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if not np.all(np.isfinite(y_pred)):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "nrmse": np.nan}
    r = rmse(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": r,
        "r2": float(r2_score(y_true, y_pred)),
        "nrmse": r / float(np.mean(y_true)),  # comparable across datasets with different currencies
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 seed: int = SEED_DATA, alpha: float = 0.05) -> dict:
    """Percentile CI of MAE / RMSE / R2 / NRMSE from resampling test rows with replacement."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    idx = np.random.default_rng(seed).integers(0, len(y_true), size=(n_boot, len(y_true)))
    yt, err = y_true[idx], (y_pred - y_true)[idx]
    rmse_b = np.sqrt((err ** 2).mean(axis=1))
    sst = ((yt - yt.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):  # tiny samples can resample a constant target
        samples = {
            "mae": np.abs(err).mean(axis=1),
            "rmse": rmse_b,
            "r2": 1.0 - (err ** 2).sum(axis=1) / sst,
            "nrmse": rmse_b / yt.mean(axis=1),
        }
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in samples.items()}
