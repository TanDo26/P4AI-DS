"""Final model: retrain on the full training set with the tuned hyperparameters and
n_epochs = median(best_epoch over CV folds), then evaluate on the test set exactly once.

This is the only module that reads the test split."""
import numpy as np
import pandas as pd

from .data import load_dataset
from .explain import ft_attention_importance, permutation_importance, tabnet_importance
from .metrics import bootstrap_ci, regression_metrics
from .models import get_model_module
from .preprocess import TabularPreprocessor, TargetTransformer, preprocessing_markdown
from .replot import plot_test
from .splits import load_test_idx, load_train_idx
from .utils import Timer, env_info, load_json, results_dir, save_json

MASK_ROWS = 12  # TabNet masks kept for the heatmap of sample test houses


def run_final(dataset: str, model_name: str, device: str) -> dict:
    module = get_model_module(model_name)
    out = results_dir(dataset, model_name)
    best_path = out / "best_params.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} missing - run `python run.py tune` first")
    info = load_json(best_path)
    params, n_epochs = info["params"], info["n_epochs_final"]

    X, y, spec = load_dataset(dataset)
    train_rows, test_rows = load_train_idx(dataset), load_test_idx(dataset)
    pre = TabularPreprocessor(spec, module.KIND, params["scaler"]).fit(X.iloc[train_rows])
    tt = TargetTransformer(log=True).fit(y[train_rows])
    model = module.build(params, pre, dataset, device)
    print(f"[final] {dataset}/{model_name}: training on {len(train_rows)} rows for {n_epochs} epochs", flush=True)
    with Timer() as t_fit:
        model.fit(pre.transform(X.iloc[train_rows]), tt.transform(y[train_rows]), n_epochs=n_epochs)

    X_test, y_test = X.iloc[test_rows], y[test_rows]
    with Timer() as t_pred:
        X_test_t = pre.transform(X_test)
        pred = tt.inverse(model.predict(X_test_t))
    metrics = regression_metrics(y_test, pred)
    result = {
        "dataset": dataset, "model": model_name, **metrics, "ci95": bootstrap_ci(y_test, pred),
        "fit_time": t_fit.elapsed, "predict_time": t_pred.elapsed, "n_epochs": n_epochs,
        "n_train": len(train_rows), "n_test": len(test_rows), "params": params, "env": env_info(device),
    }
    save_json(result, out / "test_metrics.json")
    pd.DataFrame({"row": test_rows, "y_true": y_test, "y_pred": pred}).to_csv(out / "predictions_test.csv", index=False)
    (out / "preprocessing.md").write_text(preprocessing_markdown(dataset, model_name, spec, params["scaler"]), encoding="utf-8")
    model.save(out / "model")

    print("[final] explanation: permutation importance on test", flush=True)
    perm = permutation_importance(lambda df: tt.inverse(model.predict(pre.transform(df))), X_test, y_test)
    perm.to_csv(out / "feature_importance_permutation.csv", index=False)
    if model_name == "ft_transformer":
        ft_attention_importance(model, X_test_t, pre.feature_names).to_csv(
            out / "feature_importance_attention.csv", index=False)
    else:
        imp, masks = tabnet_importance(model, X_test_t, pre.feature_names)
        imp.to_csv(out / "feature_importance_tabnet.csv", index=False)
        np.savez(out / "tabnet_masks.npz", feature_names=np.array(pre.feature_names),
                 **{f"step_{s}": m[:MASK_ROWS] for s, m in masks.items()})
    plot_test(dataset, model_name)

    ci = result["ci95"]
    print(f"[final] {dataset}/{model_name}: test MAE {metrics['mae']:,.0f} | RMSE {metrics['rmse']:,.0f} "
          f"[{ci['rmse'][0]:,.0f}, {ci['rmse'][1]:,.0f}] | R² {metrics['r2']:.4f} | fit {t_fit.elapsed:.1f}s", flush=True)
    return result
