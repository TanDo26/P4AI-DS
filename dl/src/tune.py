"""Optuna tuning. Objective = mean 5-fold CV RMSE on the original price scale (training data only).
The study lives in results/<dataset>/<model>/optuna.db, so an interrupted run resumes where it stopped.
Each completed trial stores its per-fold metrics and learning curves, so export needs no extra CV run."""
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from .config import SEED_DATA, SEED_MODEL
from .cv import run_cv, summarize
from .models import get_model_module
from .replot import plot_tuning
from .utils import Timer, env_info, results_dir, save_json


def load_study(dataset: str, model_name: str) -> optuna.Study:
    db = results_dir(dataset, model_name) / "optuna.db"
    return optuna.create_study(
        study_name=f"{dataset}-{model_name}", storage=f"sqlite:///{db.as_posix()}", load_if_exists=True,
        direction="minimize", sampler=TPESampler(seed=SEED_DATA),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )


def tune(dataset: str, model_name: str, n_trials: int, device: str, timeout: float | None = None) -> None:
    """Run until the study holds n_trials finished (complete + pruned) trials, then export results."""
    module = get_model_module(model_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = load_study(dataset, model_name)
    if not study.trials:
        study.enqueue_trial(module.default_trial_params(dataset))  # trial 0 = paper/library default config
    finished = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials)
    remaining = max(0, n_trials - finished)
    print(f"[tune] {dataset}/{model_name}: {finished} trials done, running {remaining} more on {device}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = module.suggest_params(trial, dataset)
        trial.set_user_attr("params", params)
        with Timer() as t:
            res = run_cv(dataset, model_name, params, device=device, trial=trial)
        s = res["summary"]
        for key in ("mae_mean", "rmse_std", "r2_mean", "n_epochs_final", "best_epochs"):
            trial.set_user_attr(key, s[key])
        trial.set_user_attr("cv_seconds", t.elapsed)
        trial.set_user_attr("folds", res["folds"].to_dict("records"))
        trial.set_user_attr("histories", res["histories"])
        if not np.isfinite(s["rmse_mean"]):
            raise optuna.TrialPruned()
        return s["rmse_mean"]

    def log(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == TrialState.COMPLETE:
            value = f"cv_rmse={trial.value:>12,.0f}"
        else:  # pruned: only a running mean over the first folds exists
            value = f"stopped after fold {len(trial.intermediate_values)}/5"
        best = f"{study.best_value:,.0f}" if any(t.state == TrialState.COMPLETE for t in study.trials) else "-"
        print(f"  trial {trial.number:3d} {trial.state.name:8s} {value:28s} best={best}", flush=True)

    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout, gc_after_trial=True, callbacks=[log],
                       catch=(RuntimeError, ValueError))
    export(study, dataset, model_name, device)


def export(study: optuna.Study, dataset: str, model_name: str, device: str) -> None:
    """Save all trials, the best trial's folds and learning curves, best_params.json and figures."""
    out = results_dir(dataset, model_name)
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials["state"] = trials["state"].astype(str).str.replace("TrialState.", "", regex=False)
    trials.drop(columns=["user_attrs_folds", "user_attrs_histories"], errors="ignore").to_csv(
        out / "optuna_trials.csv", index=False)
    best = study.best_trial
    params = best.user_attrs["params"]
    folds = pd.DataFrame(best.user_attrs["folds"])
    histories = best.user_attrs["histories"]
    folds.to_csv(out / "cv_folds.csv", index=False)
    save_json(histories, out / "cv_histories.json")
    s = summarize(folds)

    counts = {state.name.lower(): sum(t.state == state for t in study.trials)
              for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
    save_json({
        "dataset": dataset,
        "model": model_name,
        "params": params,
        "n_epochs_final": s["n_epochs_final"],
        "cv": s,
        "optuna": {"best_trial": best.number, "best_value": best.value, "trials": counts,
                   "sampler": f"TPESampler(seed={SEED_DATA})", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)"},
        "preprocessing": {"use_fe": True, "target": "log1p + standardize", "scaler": params["scaler"]},
        "seeds": {"data": SEED_DATA, "model": SEED_MODEL},
        "env": env_info(device),
    }, out / "best_params.json")

    plot_tuning(dataset, model_name, study)
    print(f"[tune] {dataset}/{model_name}: CV RMSE {s['rmse_mean']:,.0f} ± {s['rmse_std']:,.0f}, "
          f"R² {s['r2_mean']:.4f}, final epochs = {s['n_epochs_final']}", flush=True)
