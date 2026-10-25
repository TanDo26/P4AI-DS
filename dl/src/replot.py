"""Per-model figures rebuilt from saved artifacts only (no training, no split access).
Called at the end of tune/final, and by `python run.py replot` after changing plot styles."""
import numpy as np
import optuna
import pandas as pd

from . import plots
from .utils import load_json, model_figures_dir, results_dir

MIN_TRIALS_FOR_PARAM_IMPORTANCE = 5  # fANOVA on fewer completed trials is meaningless


def load_saved_study(dataset: str, model_name: str) -> optuna.Study:
    db = results_dir(dataset, model_name) / "optuna.db"
    return optuna.load_study(study_name=f"{dataset}-{model_name}", storage=f"sqlite:///{db.as_posix()}")


def plot_tuning(dataset: str, model_name: str, study: optuna.Study | None = None) -> None:
    out, figs = results_dir(dataset, model_name), model_figures_dir(dataset, model_name)
    color, label = plots.model_color(model_name), plots.model_label(model_name)
    trials = pd.read_csv(out / "optuna_trials.csv")
    plots.optuna_history(trials, figs / "optuna_history.png", f"{label} — Optuna ({dataset})", color)

    importance_png = figs / "optuna_param_importance.png"
    if (trials["state"] == "COMPLETE").sum() >= MIN_TRIALS_FOR_PARAM_IMPORTANCE:
        try:
            importances = optuna.importance.get_param_importances(study or load_saved_study(dataset, model_name))
            plots.param_importance(importances, importance_png, f"{label} — độ quan trọng hyperparameter ({dataset})", color)
        except (RuntimeError, ValueError) as e:
            print(f"[plots] param importance skipped: {e}", flush=True)
    else:
        importance_png.unlink(missing_ok=True)

    folds, histories = pd.read_csv(out / "cv_folds.csv"), load_json(out / "cv_histories.json")
    plots.learning_curve(histories[0], int(folds["best_epoch"].iloc[0]), figs / "learning_curve_fold0.png",
                         f"{label} — learning curve, fold 0 ({dataset})")


def plot_test(dataset: str, model_name: str) -> None:
    out, figs = results_dir(dataset, model_name), model_figures_dir(dataset, model_name)
    color, label = plots.model_color(model_name), plots.model_label(model_name)
    pred = pd.read_csv(out / "predictions_test.csv")
    y, y_hat = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    plots.pred_vs_actual(y, y_hat, figs / "pred_vs_actual.png", f"{label} — dự đoán vs thực tế, test ({dataset})", color)
    plots.residual_hist(y, y_hat, figs / "residuals.png", f"{label} — phân bố sai số, test ({dataset})", color)

    perm = pd.read_csv(out / "feature_importance_permutation.csv")
    plots.importance_bars(perm, "rmse_increase_mean", figs / "importance_permutation.png",
                          f"{label} — permutation importance ({dataset})", "Mức tăng RMSE khi xáo trộn feature (giá gốc)",
                          color, err="rmse_increase_std", money=True)
    if model_name == "ft_transformer":
        att = pd.read_csv(out / "feature_importance_attention.csv")
        plots.importance_bars(att, "attention", figs / "importance_attention.png",
                              f"{label} — attention [CLS] → feature ({dataset})", "Attention trung bình (tổng = 1)", color)
    else:
        imp = pd.read_csv(out / "feature_importance_tabnet.csv")
        plots.importance_bars(imp, "importance_test", figs / "importance_tabnet.png",
                              f"{label} — feature importance từ mask ({dataset})", "Importance tổng hợp (tổng = 1)", color)
        masks_file = out / "tabnet_masks.npz"
        if masks_file.exists():
            saved = np.load(masks_file)
            masks = {int(k.split("_")[1]): saved[k] for k in saved.files if k.startswith("step_")}
            plots.tabnet_masks(masks, list(saved["feature_names"]), figs / "tabnet_masks.png",
                               f"{label} — mask từng bước quyết định, 12 căn nhà test đầu tiên ({dataset})")


def replot_all() -> None:
    from .config import DATASETS, MODEL_NAMES
    for dataset in DATASETS:
        for model_name in MODEL_NAMES:
            out = results_dir(dataset, model_name)
            if (out / "cv_histories.json").exists():
                plot_tuning(dataset, model_name)
            if (out / "predictions_test.csv").exists():
                plot_test(dataset, model_name)
                print(f"[replot] {dataset}/{model_name}", flush=True)
