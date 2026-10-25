"""End-to-end run of every command with very short training, writing into a temporary folder."""
import json

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
                  "figures/optuna_history.png", "figures/importance_permutation.png"):
            assert (out / f).exists(), f
        metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
        assert metrics["n_test"] == 109 and metrics["ci95"]["rmse"][0] <= metrics["rmse"] <= metrics["ci95"]["rmse"][1]
    assert (results / "housing" / "ft_transformer" / "figures" / "importance_attention.png").exists()
    assert (results / "housing" / "tabnet" / "figures" / "tabnet_masks.png").exists()

    df = summary.build_summary()
    assert len(df) == 2 and (results / "summary.md").exists()
    assert (figures / "summary" / "cv_rmse.png").exists()


def test_ablation_and_size_curve(tmp_outputs):
    results, figures = tmp_outputs
    df = ablation.run_ablation("housing", list(MODEL_NAMES), "cpu")
    assert len(df) == 2 * len(ablation.VARIANTS) and (figures / "housing" / "ablation.png").exists()
    df = size_curve.run_size_curve(list(MODEL_NAMES), "cpu", sizes=(436,), include_full=False)
    assert set(df["n_fold_train"]) == {436} and (figures / "kc" / "size_curve.png").exists()
