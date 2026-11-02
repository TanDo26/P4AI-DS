"""End-to-end run of every command with very short training, writing into a temporary folder."""
import json

import pandas as pd
import pytest

from experiments import ablation, size_curve
from src import summary, utils
from src.config import MODEL_NAMES
from src.final import run_final
from src.models import ft_transformer, mlp, tabnet
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
    monkeypatch.setattr(mlp, "MAX_EPOCHS", 3)
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
        # final model: 85% fit / 15% early stopping (inner_split of the train set)
        assert (metrics["n_fit"], metrics["n_val"]) == (370, 66) and 1 <= metrics["n_epochs"] <= 3
    assert (results / "housing" / "ft_transformer" / "figures" / "importance_attention.png").exists()
    assert (results / "housing" / "tabnet" / "figures" / "tabnet_masks.png").exists()

    # the MLP's tuning runs its fixed configurations in order (2 of the 5 here)
    trials = pd.read_csv(results / "housing" / "mlp" / "optuna_trials.csv")
    assert list(trials["params_config"]) == list(mlp.CONFIGS)[:2]

    df = summary.build_summary()
    assert len(df) == 3 and "| housing | MLP | 2 cấu hình cố định |" in (results / "summary.md").read_text(encoding="utf-8")
    assert (figures / "summary" / "cv_rmse.png").exists()


def test_ablation_and_size_curve(tmp_outputs):
    results, figures = tmp_outputs
    df = ablation.run_ablation("housing", list(MODEL_NAMES), "cpu")
    # the MLP skips "StandardScaler": its baseline already uses it
    assert len(df) == 3 * len(ablation.VARIANTS) - 1 and (figures / "housing" / "ablation.png").exists()
    df = size_curve.run_size_curve(list(MODEL_NAMES), "cpu", sizes=(436,), include_full=False)
    assert set(df["n_fold_train"]) == {436} and (figures / "kc" / "size_curve.png").exists()
