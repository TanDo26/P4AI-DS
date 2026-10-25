"""Collect every model x dataset result into one table (+ optional ML-team results) and comparison figures.

The headline metrics are never altered. The error analysis (how much of the squared error comes from the
worst test houses, and metrics without the single worst house) is reported next to them, for analysis only."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import DATASETS, MODEL_NAMES, RESULTS_DIR
from .data import load_raw
from .metrics import regression_metrics
from .utils import figures_dir, load_json


def error_analysis(dataset: str, model: str) -> dict:
    """Concentration of squared error in the worst test rows; writes largest_errors_test.csv (top 10)."""
    d = RESULTS_DIR / dataset / model
    p = pd.read_csv(d / "predictions_test.csv")
    err2 = (p["y_pred"] - p["y_true"]) ** 2
    order = err2.sort_values(ascending=False).index
    top = p.loc[order[:10]].assign(ratio=lambda t: t["y_pred"] / t["y_true"])
    top.merge(load_raw(dataset).drop(columns=["price"]), left_on="row", right_index=True).to_csv(
        d / "largest_errors_test.csv", index=False)
    rest = p.drop(index=order[:1])
    without = regression_metrics(rest["y_true"].to_numpy(), rest["y_pred"].to_numpy())
    return {"sse_share_top1": float(err2[order[0]] / err2.sum()),
            "sse_share_top5": float(err2[order[:5]].sum() / err2.sum()),
            "max_pred_ratio": float(np.max(p["y_pred"] / p["y_true"])),
            "test_rmse_without_top1": without["rmse"], "test_r2_without_top1": without["r2"]}


def collect() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            d = RESULTS_DIR / dataset / model
            if not (d / "best_params.json").exists() or not (d / "test_metrics.json").exists():
                continue
            info, test = load_json(d / "best_params.json"), load_json(d / "test_metrics.json")
            cv = info["cv"]
            row = {"dataset": dataset, "model": model, "n_trials": info["optuna"]["trials"]["complete"],
                   "n_pruned": info["optuna"]["trials"]["pruned"]}
            for m in ("mae", "rmse", "r2", "nrmse"):
                row[f"cv_{m}_mean"], row[f"cv_{m}_std"] = cv[f"{m}_mean"], cv[f"{m}_std"]
            for m in ("mae", "rmse", "r2", "nrmse"):
                row[f"test_{m}"] = test[m]
                row[f"test_{m}_ci_lo"], row[f"test_{m}_ci_hi"] = test["ci95"][m]
            row.update(fit_time=test["fit_time"], predict_time=test["predict_time"], n_epochs=test["n_epochs"],
                       device=test["env"]["device"], hardware=test["env"]["hardware"])
            row.update(error_analysis(dataset, model))
            rows.append(row)
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame) -> str:
    def money(v):
        return f"{v:,.0f}"

    lines = ["| Dataset | Model | Trials | CV RMSE | CV R² | Test MAE | Test RMSE [95% CI] | Test R² [95% CI] | NRMSE | Fit (s) | Predict (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        trials = "1 (mặc định, chưa tune)" if r["n_trials"] + r["n_pruned"] == 1 else f"{r['n_trials']} + {r['n_pruned']} pruned"
        lines.append(
            f"| {r['dataset']} | {plots.model_label(r['model'])} | {trials} | {money(r['cv_rmse_mean'])} ± {money(r['cv_rmse_std'])} "
            f"| {r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f} | {money(r['test_mae'])} "
            f"| {money(r['test_rmse'])} [{money(r['test_rmse_ci_lo'])}, {money(r['test_rmse_ci_hi'])}] "
            f"| {r['test_r2']:.4f} [{r['test_r2_ci_lo']:.3f}, {r['test_r2_ci_hi']:.3f}] | {r['test_nrmse']:.3f} "
            f"| {r['fit_time']:.1f} | {r['predict_time']:.2f} |")
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
        print("[summary] no finished model yet (need best_params.json + test_metrics.json)")
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
