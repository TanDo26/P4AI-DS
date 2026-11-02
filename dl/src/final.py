"""Final model: train with the tuned hyperparameters on 85% of the training set, early stopping on the
other 15% (inner_split, seed 42 - the protocol of the team's original MLP code), then evaluate on the test
set exactly once.

Explanations (permutation importance for every model, plus attention / TabNet masks) use that 15% validation
split and are computed before the test split is read. This is the only module that reads it."""
import numpy as np
import pandas as pd

from .data import load_dataset
from .explain import ft_attention_importance, permutation_importance, tabnet_importance
from .metrics import bootstrap_ci, regression_metrics
from .models import get_model_module
from .preprocess import TabularPreprocessor, TargetTransformer, preprocessing_markdown
from .replot import plot_test
from .splits import inner_split, load_test_idx, load_train_idx
from .utils import Timer, env_info, load_json, results_dir, save_json

MASK_ROWS = 12  # TabNet masks kept for the heatmap of sample validation houses


def run_final(dataset: str, model_name: str, device: str) -> dict:
    module = get_model_module(model_name)
    out = results_dir(dataset, model_name)
    best_path = out / "best_params.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} missing - run `python run.py tune` first")
    params = load_json(best_path)["params"]

    X, y, spec = load_dataset(dataset)
    train_rows = load_train_idx(dataset)
    fit_rows, val_rows = inner_split(train_rows)
    pre = TabularPreprocessor(spec, module.KIND, params["scaler"]).fit(X.iloc[fit_rows])
    tt = TargetTransformer(log=True).fit(y[fit_rows])
    model = module.build(params, pre, dataset, device)
    X_val, y_val = X.iloc[val_rows], y[val_rows]
    X_val_t = pre.transform(X_val)
    print(f"[final] {dataset}/{model_name}: training on {len(fit_rows)} rows, early stopping on {len(val_rows)}",
          flush=True)
    with Timer() as t_fit:
        model.fit(pre.transform(X.iloc[fit_rows]), tt.transform(y[fit_rows]), X_val_t, tt.transform(y_val))
    n_epochs = model.best_epoch_
    save_json(model.history_, out / "final_history.json")
    (out / "preprocessing.md").write_text(preprocessing_markdown(dataset, model_name, spec, params["scaler"]), encoding="utf-8")
    model.save(out / "model")

    print(f"[final] explanation on the {len(val_rows)}-row validation split", flush=True)
    perm = permutation_importance(lambda df: tt.inverse(model.predict(pre.transform(df))), X_val, y_val)
    perm.to_csv(out / "feature_importance_permutation.csv", index=False)
    if model_name == "ft_transformer":
        ft_attention_importance(model, X_val_t, pre.feature_names).to_csv(
            out / "feature_importance_attention.csv", index=False)
    elif model_name == "tabnet":
        imp, masks = tabnet_importance(model, X_val_t, pre.feature_names)
        imp.to_csv(out / "feature_importance_tabnet.csv", index=False)
        np.savez(out / "tabnet_masks.npz", feature_names=np.array(pre.feature_names),
                 **{f"step_{s}": m[:MASK_ROWS] for s, m in masks.items()})

    test_rows = load_test_idx(dataset)
    X_test, y_test = X.iloc[test_rows], y[test_rows]
    with Timer() as t_pred:
        pred = tt.inverse(model.predict(pre.transform(X_test)))
    metrics = regression_metrics(y_test, pred)
    result = {
        "dataset": dataset, "model": model_name, **metrics, "ci95": bootstrap_ci(y_test, pred),
        "fit_time": t_fit.elapsed, "predict_time": t_pred.elapsed, "n_epochs": n_epochs,
        "n_train": len(train_rows), "n_fit": len(fit_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "params": params, "env": env_info(device),
    }
    save_json(result, out / "test_metrics.json")
    pd.DataFrame({"row": test_rows, "y_true": y_test, "y_pred": pred}).to_csv(out / "predictions_test.csv", index=False)
    plot_test(dataset, model_name)

    ci = result["ci95"]
    print(f"[final] {dataset}/{model_name}: test MAE {metrics['mae']:,.0f} | RMSE {metrics['rmse']:,.0f} "
          f"[{ci['rmse'][0]:,.0f}, {ci['rmse'][1]:,.0f}] | R² {metrics['r2']:.4f} | fit {t_fit.elapsed:.1f}s, "
          f"best epoch {n_epochs}", flush=True)
    return result
