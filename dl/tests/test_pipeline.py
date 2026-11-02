"""End-to-end run of every command with very short training, writing into a temporary folder."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments import ablation, size_curve
from src import summary, utils
from src.config import MODEL_NAMES
from src.final import run_final
from src.models import ft_transformer, tabnet
from src.splits import make_splits
from src.tune import tune


@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    results, figures = tmp_path / "results", tmp_path / "figures"
    for module in (utils, summary, ablation, size_curve):
        if hasattr(module, "RESULTS_DIR"):
            monkeypatch.setattr(module, "RESULTS_DIR", results)
    monkeypatch.setattr(utils, "FIGURES_DIR", figures)
    monkeypatch.setattr(summary, "MLP_RESULTS_DIR", tmp_path / "results_mlp")
    monkeypatch.setattr(ft_transformer, "train_config", lambda n: {"batch_size": 64, "max_epochs": 3, "patience": 2})
    monkeypatch.setattr(tabnet, "MAX_EPOCHS", 3)
    for ds in ("housing", "kc"):
        make_splits(ds)
    return results, figures


def test_tune_final_summary(tmp_outputs):
    results, figures = tmp_outputs
    for model in MODEL_NAMES:
        tune("housing", model, n_trials=2, device="cpu")
        run_final("housing", model, "cpu")
        out = results / "housing" / model
        for f in ("best_params.json", "optuna_trials.csv", "cv_folds.csv", "cv_histories.json", "test_metrics.json",
                  "predictions_test.csv", "preprocessing.md", "feature_importance_permutation.csv",
                  "figures/pred_vs_actual.png", "figures/residuals.png", "figures/learning_curve_fold0.png",
                  "figures/optuna_history.png", "figures/importance_permutation.png", "final_history.json"):
            assert (out / f).exists(), f
        metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
        assert metrics["n_test"] == 109 and metrics["ci95"]["rmse"][0] <= metrics["rmse"] <= metrics["ci95"]["rmse"][1]
        # final model: 85% fit / 15% early stopping, the same split as the MLP
        assert (metrics["n_fit"], metrics["n_val"]) == (370, 66) and 1 <= metrics["n_epochs"] <= 3
    assert (results / "housing" / "ft_transformer" / "figures" / "importance_attention.png").exists()
    assert (results / "housing" / "tabnet" / "figures" / "tabnet_masks.png").exists()

    df = summary.build_summary()
    assert len(df) == 2 and (results / "summary.md").exists()
    assert (figures / "summary" / "cv_rmse.png").exists()

    # MLP output in the dl_mlp_experiment.py format: housing is added, a KC run with the old CV seed is skipped
    mlp = summary.MLP_RESULTS_DIR
    preds = pd.read_csv(results / "housing" / "ft_transformer" / "predictions_test.csv")
    d = mlp / "housing_prices"
    d.mkdir(parents=True)
    (d / "best_config_and_metrics.json").write_text(json.dumps(
        {"split": {"random_state": 42}, "cv": {"random_state": 42}, "device": "cpu", "final_fit_time_sec": 2.0,
         "selected_hyperparameters": {"name": "mlp_b", "final_best_epoch": 7}}), encoding="utf-8")
    pd.DataFrame({"config": ["mlp_a", "mlp_b"]}).to_csv(d / "cv_results.csv", index=False)
    folds = pd.DataFrame({"config": ["mlp_a"] * 5 + ["mlp_b"] * 5, "fold": [1, 2, 3, 4, 5] * 2,
                          "MAE": 1.0, "RMSE": [9e5] * 5 + [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6], "R2": 0.5})
    folds.to_csv(d / "cv_fold_details.csv", index=False)
    pd.DataFrame({"source_row_index": preds["row"], "actual_price": preds["y_true"],
                  "predicted_price": preds["y_true"] * 1.1}).to_csv(d / "test_predictions.csv", index=False)
    (mlp / "king_county").mkdir()
    (mlp / "king_county" / "best_config_and_metrics.json").write_text(
        json.dumps({"split": {"random_state": 42}, "cv": {"random_state": 36}}), encoding="utf-8")

    df = summary.build_summary()
    r = df[df["model"] == "mlp"].iloc[0]
    assert len(df) == 3 and r["dataset"] == "housing" and r["n_trials"] == 2 and r["n_epochs"] == 7
    assert np.isclose(r["cv_rmse_mean"], 1.2e6) and np.isfinite(r["cv_nrmse_mean"]) and np.isnan(r["predict_time"])
    assert np.isclose(r["test_rmse"], np.sqrt(np.mean((0.1 * preds["y_true"]) ** 2)))
    assert (d / "largest_errors_test.csv").exists()
    assert "| housing | MLP | 2 cấu hình cố định |" in (results / "summary.md").read_text(encoding="utf-8")


def test_ablation_and_size_curve(tmp_outputs):
    results, figures = tmp_outputs
    df = ablation.run_ablation("housing", list(MODEL_NAMES), "cpu")
    assert len(df) == 2 * len(ablation.VARIANTS) and (figures / "housing" / "ablation.png").exists()
    df = size_curve.run_size_curve(list(MODEL_NAMES), "cpu", sizes=(436,), include_full=False)
    assert set(df["n_fold_train"]) == {436} and (figures / "kc" / "size_curve.png").exists()
