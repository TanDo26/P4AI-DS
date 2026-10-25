"""5-fold CV on the training set. Per fold: inner split -> fit preprocessing on inner_train ->
train with early stopping on inner_val -> score fold_val on the original price scale."""
import numpy as np
import optuna
import pandas as pd

from .config import N_FOLDS
from .data import load_dataset
from .metrics import regression_metrics
from .models import get_model_module
from .preprocess import TabularPreprocessor, TargetTransformer
from .splits import fold_indices, inner_split, load_folds, subset_rows
from .utils import Timer


def summarize(folds: pd.DataFrame) -> dict:
    out = {}
    for m in ("mae", "rmse", "r2", "nrmse"):
        out[f"{m}_mean"] = float(folds[m].mean())
        out[f"{m}_std"] = float(folds[m].std(ddof=1))
    out["best_epochs"] = [int(e) for e in folds["best_epoch"]]
    out["n_epochs_final"] = max(1, int(round(float(np.median(folds["best_epoch"])))))
    out["fit_time_total"] = float(folds["fit_time"].sum())
    return out


def run_cv(dataset: str, model_name: str, params: dict, *, device: str = "cpu", use_fe: bool = True,
           target_log: bool = True, train_size: int | None = None, trial: optuna.Trial | None = None,
           verbose: bool = False) -> dict:
    """train_size: learning curve - use the first `train_size` rows of each fold's fixed permutation."""
    module = get_model_module(model_name)
    X, y, spec = load_dataset(dataset, use_fe=use_fe)
    folds = load_folds(dataset)

    records, histories = [], []
    for k in range(N_FOLDS):
        fit_rows, val_rows = fold_indices(folds, k)
        if train_size is not None:
            fit_rows = subset_rows(dataset, k, train_size)
        inner_tr, inner_val = inner_split(fit_rows)

        pre = TabularPreprocessor(spec, module.KIND, params["scaler"]).fit(X.iloc[inner_tr])
        tt = TargetTransformer(log=target_log).fit(y[inner_tr])
        model = module.build(params, pre, dataset, device)
        with Timer() as t:
            model.fit(pre.transform(X.iloc[inner_tr]), tt.transform(y[inner_tr]),
                      pre.transform(X.iloc[inner_val]), tt.transform(y[inner_val]))
        pred = tt.inverse(model.predict(pre.transform(X.iloc[val_rows])))

        rec = {"fold": k, "n_fold_train": len(fit_rows), "n_inner_train": len(inner_tr), "n_inner_val": len(inner_val),
               "n_val": len(val_rows), "best_epoch": model.best_epoch_, "fit_time": t.elapsed,
               **regression_metrics(y[val_rows], pred)}
        records.append(rec)
        histories.append(model.history_)
        if verbose:
            print(f"    fold {k}: rmse={rec['rmse']:,.0f} r2={rec['r2']:.4f} best_epoch={rec['best_epoch']} "
                  f"({t.elapsed:.1f}s)", flush=True)
        if trial is not None:
            running = float(np.mean([r["rmse"] for r in records]))
            trial.report(running if np.isfinite(running) else float("inf"), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    df = pd.DataFrame(records)
    return {"folds": df, "histories": histories, "summary": summarize(df)}
