"""Fast end-to-end checks of the building blocks on a few hundred Housing rows."""
import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from optuna.trial import TrialState

from src import utils
from src.data import load_dataset
from src.explain import ft_attention_importance, permutation_importance, tabnet_importance
from src.metrics import bootstrap_ci, regression_metrics
from src.models import ft_transformer, get_model_module, mlp, modernnca, tabm, tabnet
from src.preprocess import TabularPreprocessor, TargetTransformer
from src.splits import load_train_idx, make_splits
from src.tune import archive_mlp_grid, enqueue_fixed_trials

SHORT = {"batch_size": 32, "max_epochs": 5, "patience": 2}


@pytest.fixture(scope="module")
def housing():
    make_splits("housing")
    X, y, spec = load_dataset("housing")
    rows = load_train_idx("housing")
    return X, y, spec, rows[:200], rows[200:260]


@pytest.fixture(autouse=True)
def short_training(monkeypatch):
    for module in (ft_transformer, tabm, modernnca):
        monkeypatch.setattr(module, "train_config", lambda n: SHORT)
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
    assert sorted(waiting, key=str) == sorted(fixed, key=str)


@pytest.mark.parametrize("dataset", ["housing", "kc"])
@pytest.mark.parametrize("model_name", ["mlp", "tabm", "modernnca"])
def test_fixed_trials_lie_inside_the_search_space(model_name, dataset):
    """Optuna samples a random value instead of a queued one that is outside the space; this must not happen."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    module = get_model_module(model_name)
    expected = ([{**c, "scaler": "standard"} for c in mlp.CONFIGS.values()] if model_name == "mlp"
                else [module.default_params(dataset)])
    study = optuna.create_study()
    for fixed in module.fixed_trials(dataset):
        study.enqueue_trial(fixed)
    got = [module.suggest_params(study.ask(), dataset) for _ in expected]
    assert got == expected


def test_former_mlp_grid_is_archived(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_DIR", tmp_path)
    out = utils.results_dir("kc", "mlp")
    storage = optuna.storages.RDBStorage(f"sqlite:///{(out / 'optuna.db').as_posix()}")

    def grid_trial(trial):
        trial.suggest_categorical("config", list(mlp.CONFIGS))
        return 0.0

    optuna.create_study(study_name="kc-mlp", storage=storage).optimize(grid_trial, n_trials=1)
    storage.engine.dispose()  # Windows cannot move a file that an open connection holds
    (out / "test_metrics.json").write_text("{}")
    pd.DataFrame({"model": ["mlp", "ft_transformer"], "rmse": [1.0, 2.0]}).to_csv(tmp_path / "kc" / "size_curve.csv")

    archive_mlp_grid("kc")
    grid = tmp_path / "kc" / "mlp_grid"
    assert not out.exists() and (grid / "optuna.db").exists() and (grid / "test_metrics.json").exists()
    assert list(pd.read_csv(tmp_path / "kc" / "size_curve.csv")["model"]) == ["ft_transformer"]
    assert list(pd.read_csv(grid / "size_curve.csv")["model"]) == ["mlp"]
    archive_mlp_grid("kc")  # the new (empty) mlp folder is left alone
    assert grid.exists()


@pytest.mark.parametrize("model_name", ["ft_transformer", "tabnet", "mlp", "tabm", "modernnca"])
def test_model_fit_predict(housing, model_name):
    X, y, spec, tr, val = housing
    module = get_model_module(model_name)
    params = module.default_params("housing")
    if model_name == "ft_transformer":
        params = {**params, "n_blocks": 1, "d_block": 32}
    elif model_name == "tabm":
        params = {**params, "n_blocks": 1, "d_block": 64}
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


def _fit(housing, model_name, params):
    X, y, spec, tr, val = housing
    module = get_model_module(model_name)
    pre = TabularPreprocessor(spec, module.KIND, params["scaler"]).fit(X.iloc[tr])
    tt = TargetTransformer().fit(y[tr])
    model = module.build(params, pre, "housing", "cpu")
    model.fit(pre.transform(X.iloc[tr]), tt.transform(y[tr]), pre.transform(X.iloc[val]), tt.transform(y[val]))
    return model, pre.transform(X.iloc[val])


def test_tabm_predicts_the_mean_of_its_members(housing):
    model, x_val = _fit(housing, "tabm", {**tabm.default_params("housing"), "n_blocks": 1, "d_block": 64})
    model.net_.eval()
    with torch.no_grad():
        members = model.net_(torch.as_tensor(x_val)).squeeze(-1)
    assert members.shape == (len(x_val), tabm.K)
    assert np.allclose(model.predict(x_val), members.mean(1).numpy(), atol=1e-6)


@pytest.mark.parametrize("n_blocks", [0, 1])
def test_modernnca_retrieves_only_training_rows(housing, n_blocks):
    _, _, _, tr, _ = housing
    params = modernnca.default_params("housing")
    if n_blocks:  # MLP block with BatchNorm, on top of the default encoder
        params = {**params, "n_blocks": 1, "d_block": 64, "dropout": 0.1}
    model, x_val = _fit(housing, "modernnca", params)
    weights = model.neighbor_weights(x_val)
    assert model.cand_x_.shape[0] == len(tr) and weights.shape == (len(x_val), len(tr))
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(model.predict(x_val), weights @ model.cand_y_.numpy(), atol=1e-5)
