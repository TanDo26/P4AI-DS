"""Collect every model x dataset result into one table (+ optional ML-team results) and comparison figures.

The MLP comes from dl_mlp_project/results_mlp/: its test metrics, CI and NRMSE are recomputed here from its
test_predictions.csv, so every model is scored by the same code on the same test rows.

The headline metrics are never altered. The error analysis (how much of the squared error comes from the
worst test houses, and metrics without the single worst house) is reported next to them, for analysis only."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import DATASETS, MLP_DATASET_DIRS, MLP_RESULTS_DIR, MODEL_NAMES, RESULTS_DIR, SEED_DATA
from .data import load_raw
from .metrics import bootstrap_ci, regression_metrics
from .splits import load_folds
from .utils import figures_dir, load_json

METRICS = ("mae", "rmse", "r2", "nrmse")


def error_analysis(pred: pd.DataFrame, dataset: str, out: Path) -> dict:
    """Concentration of squared error in the worst test rows (pred: row, y_true, y_pred);
    writes out/largest_errors_test.csv (top 10)."""
    err2 = (pred["y_pred"] - pred["y_true"]) ** 2
    order = err2.sort_values(ascending=False).index
    top = pred.loc[order[:10]].assign(ratio=lambda t: t["y_pred"] / t["y_true"])
    top.merge(load_raw(dataset).drop(columns=["price"]), left_on="row", right_index=True).to_csv(
        out / "largest_errors_test.csv", index=False)
    rest = pred.drop(index=order[:1])
    without = regression_metrics(rest["y_true"].to_numpy(), rest["y_pred"].to_numpy())
    return {"sse_share_top1": float(err2[order[0]] / err2.sum()),
            "sse_share_top5": float(err2[order[:5]].sum() / err2.sum()),
            "max_pred_ratio": float(np.max(pred["y_pred"] / pred["y_true"])),
            "test_rmse_without_top1": without["rmse"], "test_r2_without_top1": without["r2"]}


def _metric_columns(cv: dict, test: dict, ci95: dict) -> dict:
    row = {}
    for m in METRICS:
        row[f"cv_{m}_mean"], row[f"cv_{m}_std"] = cv[f"{m}_mean"], cv[f"{m}_std"]
    for m in METRICS:
        row[f"test_{m}"] = test[m]
        row[f"test_{m}_ci_lo"], row[f"test_{m}_ci_hi"] = ci95[m]
    return row


def collect_dl() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            d = RESULTS_DIR / dataset / model
            if not (d / "best_params.json").exists() or not (d / "test_metrics.json").exists():
                continue
            info, test = load_json(d / "best_params.json"), load_json(d / "test_metrics.json")
            row = {"dataset": dataset, "model": model, "n_trials": info["optuna"]["trials"]["complete"],
                   "n_pruned": info["optuna"]["trials"]["pruned"]}
            row.update(_metric_columns(info["cv"], test, test["ci95"]))
            row.update(fit_time=test["fit_time"], predict_time=test["predict_time"], n_epochs=test["n_epochs"],
                       device=test["env"]["device"], hardware=test["env"]["hardware"])
            row.update(error_analysis(pd.read_csv(d / "predictions_test.csv"), dataset, d))
            rows.append(row)
    return rows


def collect_mlp() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        d = MLP_RESULTS_DIR / MLP_DATASET_DIRS[dataset]
        if not (d / "best_config_and_metrics.json").exists():
            continue
        info = load_json(d / "best_config_and_metrics.json")
        seeds = (info["split"]["random_state"], info["cv"]["random_state"])
        if seeds != (SEED_DATA, SEED_DATA):
            print(f"[summary] skip MLP {dataset}: split/CV seeds {seeds}, not {SEED_DATA} - rerun dl_mlp_experiment.py")
            continue
        best = info["selected_hyperparameters"]["name"]
        folds = (pd.read_csv(d / "cv_fold_details.csv").query("config == @best")
                 .rename(columns={"MAE": "mae", "RMSE": "rmse", "R2": "r2"}))
        # MLP folds are numbered from 1 and hold the same rows as fold k - 1 in splits/<dataset>/folds.csv
        price, split = load_raw(dataset)["price"].to_numpy(), load_folds(dataset)
        folds["nrmse"] = [r / price[split.loc[split["fold"] == k - 1, "row"]].mean()
                          for k, r in zip(folds["fold"], folds["rmse"])]
        cv = {}
        for m in METRICS:
            cv[f"{m}_mean"], cv[f"{m}_std"] = float(folds[m].mean()), float(folds[m].std(ddof=1))

        pred = pd.read_csv(d / "test_predictions.csv").rename(
            columns={"source_row_index": "row", "actual_price": "y_true", "predicted_price": "y_pred"})
        y_true, y_pred = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
        row = {"dataset": dataset, "model": "mlp", "n_trials": len(pd.read_csv(d / "cv_results.csv")), "n_pruned": 0}
        row.update(_metric_columns(cv, regression_metrics(y_true, y_pred), bootstrap_ci(y_true, y_pred)))
        row.update(fit_time=info["final_fit_time_sec"], predict_time=np.nan,
                   n_epochs=info["selected_hyperparameters"]["final_best_epoch"], device=info["device"], hardware=np.nan)
        row.update(error_analysis(pred, dataset, d))
        rows.append(row)
    return rows


def collect() -> pd.DataFrame:
    rows = sorted(collect_dl() + collect_mlp(), key=lambda r: DATASETS.index(r["dataset"]))
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame) -> str:
    def money(v):
        return f"{v:,.0f}"

    def trials(r):
        if r["model"] == "mlp":
            return f"{r['n_trials']} cấu hình cố định"
        if r["n_trials"] + r["n_pruned"] == 1:
            return "1 (mặc định, chưa tune)"
        return f"{r['n_trials']} + {r['n_pruned']} pruned"

    lines = ["| Dataset | Model | Trials | CV RMSE | CV R² | Test MAE | Test RMSE [95% CI] | Test R² [95% CI] | NRMSE | Fit (s) | Predict (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        predict = "—" if pd.isna(r["predict_time"]) else f"{r['predict_time']:.2f}"
        lines.append(
            f"| {r['dataset']} | {plots.model_label(r['model'])} | {trials(r)} | {money(r['cv_rmse_mean'])} ± {money(r['cv_rmse_std'])} "
            f"| {r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f} | {money(r['test_mae'])} "
            f"| {money(r['test_rmse'])} [{money(r['test_rmse_ci_lo'])}, {money(r['test_rmse_ci_hi'])}] "
            f"| {r['test_r2']:.4f} [{r['test_r2_ci_lo']:.3f}, {r['test_r2_ci_hi']:.3f}] | {r['test_nrmse']:.3f} "
            f"| {r['fit_time']:.1f} | {predict} |")
    lines += ["", "**Phân tích sai số trên test** (chỉ để phân tích — metric chính là bảng trên; "
              "chi tiết 10 căn sai nhiều nhất trong `largest_errors_test.csv`)", "",
              "| Dataset | Model | % SSE từ căn sai nhất | % SSE từ 5 căn sai nhất | max ŷ/y | RMSE nếu bỏ căn sai nhất | R² nếu bỏ căn sai nhất |",
              "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['dataset']} | {plots.model_label(r['model'])} | {r['sse_share_top1']:.1%} | {r['sse_share_top5']:.1%} "
                     f"| {r['max_pred_ratio']:.2f} | {money(r['test_rmse_without_top1'])} | {r['test_r2_without_top1']:.4f} |")
    return "\n".join(lines) + "\n"


def build_summary(ml_results: Path | None = None) -> pd.DataFrame:
    df = collect()
    if df.empty:
        print("[summary] no finished model yet (need best_params.json + test_metrics.json, or results_mlp/)")
        return df
    if ml_results is not None:
        # CSV from the ML team with at least: dataset, model, test_rmse
        ml = pd.read_csv(ml_results)
        best_ml = ml.groupby("dataset")["test_rmse"].min()
        df["best_ml_test_rmse"] = df["dataset"].map(best_ml)
        df["gap_vs_best_ml"] = (df["test_rmse"] - df["best_ml_test_rmse"]) / df["best_ml_test_rmse"]
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text(_markdown(df), encoding="utf-8")

    figs = figures_dir("summary")
    plots.cv_comparison(df, "rmse", figs / "cv_rmse.png", "CV RMSE (mean ± std, 5 fold)", "CV RMSE (giá gốc)", money=True)
    plots.cv_comparison(df, "r2", figs / "cv_r2.png", "CV R² (mean ± std, 5 fold)", "CV R²", money=False)
    plots.time_vs_rmse(df, figs / "time_vs_rmse.png", "Đánh đổi thời gian train và độ chính xác (test)")
    print(f"[summary] {len(df)} rows -> {RESULTS_DIR / 'summary.csv'}, figures -> {figs}")
    print(_markdown(df))
    return df
