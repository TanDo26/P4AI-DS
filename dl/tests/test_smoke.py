"""Fast end-to-end checks of the building blocks on a few hundred Housing rows."""
import numpy as np
import optuna
import pytest
from optuna.trial import TrialState

from src.data import load_dataset
from src.explain import ft_attention_importance, permutation_importance, tabnet_importance
from src.metrics import bootstrap_ci, regression_metrics
from src.models import ft_transformer, get_model_module, mlp, tabnet
from src.preprocess import TabularPreprocessor, TargetTransformer
from src.splits import load_train_idx, make_splits
from src.tune import enqueue_fixed_trials


@pytest.fixture(scope="module")
def housing():
    make_splits("housing")
    X, y, spec = load_dataset("housing")
    rows = load_train_idx("housing")
    return X, y, spec, rows[:200], rows[200:260]


@pytest.fixture(autouse=True)
def short_training(monkeypatch):
    monkeypatch.setattr(ft_transformer, "train_config", lambda n: {"batch_size": 32, "max_epochs": 5, "patience": 2})
    monkeypatch.setattr(tabnet, "MAX_EPOCHS", 5)
    monkeypatch.setattr(mlp, "MAX_EPOCHS", 5)


def test_target_roundtrip(housing):
    _, y, _, tr, _ = housing
    tt = TargetTransformer().fit(y[tr])
    assert np.allclose(tt.inverse(tt.transform(y[tr])), y[tr], rtol=1e-5)


def test_metrics_and_bootstrap():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert regression_metrics(y, y)["rmse"] == 0 and regression_metrics(y, y)["r2"] == 1
    pred = y + np.array([0.5, -0.5, 0.2, 0.1])
    ci = bootstrap_ci(y, pred, n_boot=200)
    assert ci["rmse"][0] <= regression_metrics(y, pred)["rmse"] <= ci["rmse"][1]


def test_onehot_preprocessing_for_mlp(housing):
    X, _, spec, tr, val = housing
    pre = TabularPreprocessor(spec, "onehot", "standard").fit(X.iloc[tr])
    x = pre.transform(X.iloc[val])
    n_levels = X["furnishingstatus"].nunique()
    assert x.dtype == np.float32 and x.shape == (len(val), len(spec.num) + n_levels) == (len(val), len(pre.feature_names))
    assert np.allclose(x[:, len(spec.num):].sum(axis=1), 1.0)  # exactly one level per row


def test_fixed_trials_are_queued_again_after_an_interruption():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()
    fixed = mlp.fixed_trials("housing")
    enqueue_fixed_trials(study, fixed, len(fixed))
    assert len(study.trials) == len(mlp.CONFIGS)
    trial = study.ask()  # first configuration, interrupted
    study.tell(trial, state=TrialState.FAIL)
    enqueue_fixed_trials(study, fixed, len(fixed))
    waiting = [t.system_attrs["fixed_params"] for t in study.trials if t.state == TrialState.WAITING]
    assert sorted(w["config"] for w in waiting) == sorted(mlp.CONFIGS)


@pytest.mark.parametrize("model_name", ["ft_transformer", "tabnet", "mlp"])
def test_model_fit_predict(housing, model_name):
    X, y, spec, tr, val = housing
    module = get_model_module(model_name)
    params = module.default_params("housing")
    if model_name == "ft_transformer":
        params = {**params, "n_blocks": 1, "d_block": 32}
    pre = TabularPreprocessor(spec, module.KIND, params["scaler"]).fit(X.iloc[tr])
    tt = TargetTransformer().fit(y[tr])
    model = module.build(params, pre, "housing", "cpu")
    model.fit(pre.transform(X.iloc[tr]), tt.transform(y[tr]), pre.transform(X.iloc[val]), tt.transform(y[val]))
    pred = tt.inverse(model.predict(pre.transform(X.iloc[val])))
    assert pred.shape == (len(val),) and np.all(np.isfinite(pred))
    assert 1 <= model.best_epoch_ <= 5 and len(model.history_["val_rmse"]) >= model.best_epoch_

    final = module.build(params, pre, "housing", "cpu").fit(pre.transform(X.iloc[tr]), tt.transform(y[tr]), n_epochs=3)
    assert final.best_epoch_ == 3 and len(final.history_["train_loss"]) == 3

    imp = permutation_importance(lambda df: tt.inverse(model.predict(pre.transform(df))), X.iloc[val], y[val], n_repeats=2)
    assert list(imp.columns) == ["feature", "rmse_increase_mean", "rmse_increase_std"] and len(imp) == X.shape[1]

    if model_name == "ft_transformer":
        att = ft_attention_importance(model, pre.transform(X.iloc[val]), pre.feature_names)
        assert np.isclose(att["attention"].sum(), 1.0) and set(att["feature"]) == set(pre.feature_names)
    elif model_name == "tabnet":
        df, masks = tabnet_importance(model, pre.transform(X.iloc[val]), pre.feature_names)
        assert np.isclose(df["importance_val"].sum(), 1.0) and len(masks) == params["n_steps"]
