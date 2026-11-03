"""Preprocessing ablation with each model's default hyperparameters (MLP: its first original configuration),
scored by 5-fold CV. The default scaler is Quantile, except Standard for the MLP (as in the original code)."""
import pandas as pd

from src import plots
from src.config import RESULTS_DIR
from src.cv import run_cv
from src.models import get_model_module
from src.utils import figures_dir

# name -> overrides of the baseline (log1p target, the model's default scaler, feature engineering on)
VARIANTS = {
    "Baseline (log1p, scaler mặc định, có FE)": {},
    "Target thô (không log)": {"target_log": False},
    "StandardScaler": {"scaler": "standard"},
    "Không feature engineering": {"use_fe": False},
}


def run_ablation(dataset: str, models: list[str], device: str) -> pd.DataFrame:
    out = RESULTS_DIR / dataset / "ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for model_name in models:
        base = get_model_module(model_name).default_params(dataset)
        for variant, v in VARIANTS.items():
            if not df.empty and ((df["model"] == model_name) & (df["variant"] == variant)).any():
                continue
            if v == {"scaler": base["scaler"]}:  # same as the baseline (the MLP already uses StandardScaler)
                continue
            params = {**base, "scaler": v.get("scaler", base["scaler"])}
            print(f"[ablation] {dataset}/{model_name}: {variant}", flush=True)
            res = run_cv(dataset, model_name, params, device=device, use_fe=v.get("use_fe", True),
                         target_log=v.get("target_log", True))
            s = res["summary"]
            row = {"model": model_name, "variant": variant,
                   **{k: s[k] for k in ("rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std")}}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(out, index=False)
    plots.ablation(df[df["model"].isin(models)], figures_dir(dataset) / "ablation.png",
                   f"Ablation preprocessing — {plots.DATASET_LABELS[dataset]} (cấu hình mặc định)")
    print(f"[ablation] -> {out}", flush=True)
    return df
