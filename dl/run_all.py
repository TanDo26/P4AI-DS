"""Run every experiment from start to finish. Run from the dl/ folder:

    python run_all.py                                   # everything, GPU if available
    python run_all.py --device cuda
    python run_all.py --datasets kc --models ft_transformer mlp
    python run_all.py --device cpu --threads 1 --datasets housing

Steps: splits -> quick tests -> for each dataset and model: tune + final model -> ablation -> size-curve
(King County) -> summary. Safe to run again after an interruption: tune resumes from optuna.db, the final
model is skipped when test_metrics.json already belongs to the best hyperparameters (--force retrains it),
ablation and size-curve only compute the rows that are missing.
"""
import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

from src.config import DATASETS, DL_DIR, MODEL_NAMES
from src.utils import configure_torch, load_json, resolve_device, results_dir

warnings.filterwarnings("ignore", message=".*ExperimentalWarning.*")


def log(message: str) -> None:
    print(f"\n[run_all {time.strftime('%H:%M:%S')}] {message}", flush=True)


def final_is_current(dataset: str, model_name: str) -> bool:
    out = results_dir(dataset, model_name)
    if not (out / "test_metrics.json").exists():
        return False
    test = load_json(out / "test_metrics.json")
    # n_val is missing from results of the older protocol (final model on 100% of train, fixed epochs)
    return "n_val" in test and test["params"] == load_json(out / "best_params.json")["params"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole DL experiment (FT-Transformer, TabNet, MLP, TabM, ModernNCA)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (1 is fastest on CPU)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--force", action="store_true", help="retrain final models that are up to date")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--ml-results", type=Path, default=None, help="CSV of the ML team: dataset, model, test_rmse")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows redirects stdout as cp1252 otherwise
    device = resolve_device(args.device)
    configure_torch(device, args.threads)
    started = time.time()

    from experiments.ablation import run_ablation
    from experiments.size_curve import run_size_curve
    from src.final import run_final
    from src.models import get_model_module
    from src.splits import make_splits
    from src.summary import build_summary
    from src.tune import tune

    log(f"device {device}; datasets {args.datasets}; models {args.models}")
    for ds in DATASETS:
        make_splits(ds)  # keeps the shared split files when they exist
    if not args.skip_tests:
        log("quick tests")
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-k", "not pipeline"], cwd=DL_DIR, check=True)

    for ds in args.datasets:
        for model in args.models:
            log(f"tune {ds}/{model}")
            tune(ds, model, get_model_module(model).n_trials(ds), device)
            if args.force or not final_is_current(ds, model):
                log(f"final {ds}/{model}")
                run_final(ds, model, device)
            else:
                log(f"final {ds}/{model}: up to date, skipped (--force to retrain)")

    for ds in args.datasets:
        log(f"ablation {ds}")
        run_ablation(ds, args.models, device)
    if "kc" in args.datasets:
        log("size-curve (King County)")
        run_size_curve(args.models, device)

    log("summary")
    build_summary(args.ml_results)
    log(f"done in {(time.time() - started) / 3600:.1f} h")


if __name__ == "__main__":
    main()
