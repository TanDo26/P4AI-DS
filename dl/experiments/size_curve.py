"""Learning curve on King County: CV score vs number of fold-train rows, hyperparameters fixed.
Isolates the effect of data size from the other differences between the two datasets."""
import pandas as pd

from src import plots
from src.config import RESULTS_DIR, SIZE_CURVE_SIZES
from src.cv import run_cv
from src.models import get_model_module
from src.utils import figures_dir, load_json

DATASET = "kc"


def run_size_curve(models: list[str], device: str, sizes: tuple[int, ...] = SIZE_CURVE_SIZES,
                   include_full: bool = True) -> pd.DataFrame:
    out = RESULTS_DIR / DATASET / "size_curve.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for model_name in models:
        module = get_model_module(model_name)
        best = RESULTS_DIR / DATASET / model_name / "best_params.json"
        if best.exists():
            params, source = load_json(best)["params"], "tuned"
        else:
            params, source = module.default_params(DATASET), "default"
            print(f"[size-curve] {model_name}: no tuned params for {DATASET}, using defaults", flush=True)
        if not df.empty:  # rows computed with other hyperparameters (e.g. defaults before tuning) are stale
            df = df[(df["model"] != model_name) | (df["params_source"] == source)]
        for n in list(sizes) + (["full"] if include_full else []):
            if not df.empty and ((df["model"] == model_name) & (df["n_requested"].astype(str) == str(n))).any():
                continue  # resume: already computed
            train_size = None if n == "full" else int(n)
            p = module.adapt_to_size(params, train_size or 10 ** 9)
            print(f"[size-curve] {model_name}: n = {n}", flush=True)
            res = run_cv(DATASET, model_name, p, device=device, train_size=train_size, verbose=True)
            part = res["folds"].assign(model=model_name, n_requested=str(n), params_source=source)
            df = pd.concat([df, part], ignore_index=True)
            df.to_csv(out, index=False)
    plots.size_curve(df[df["model"].isin(models)], figures_dir(DATASET) / "size_curve.png",
                     "King County — learning curve theo số dòng train (CV, hyperparams cố định)")
    print(f"[size-curve] -> {out}", flush=True)
    return df
