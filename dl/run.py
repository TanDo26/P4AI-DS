"""Command-line entry point. Run from the dl/ folder, e.g.

    python run.py splits
    python run.py tune  --dataset housing --model ft_transformer --n-trials 50
    python run.py final --dataset housing --model ft_transformer
    python run.py all   --dataset housing                      # tune + final, both models
    python run.py ablation --dataset kc
    python run.py size-curve
    python run.py summary [--ml-results ml.csv]
    python run.py replot                                       # figures from saved results only
"""
import argparse
import sys
import warnings
from pathlib import Path

from src.config import DATASETS, DEFAULT_N_TRIALS, MODEL_NAMES
from src.utils import configure_torch, resolve_device

warnings.filterwarnings("ignore", message=".*ExperimentalWarning.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="DL house-price experiments (FT-Transformer, TabNet)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("splits", help="create shared train/test split, folds and learning-curve subsets")
    p.add_argument("--overwrite", action="store_true")

    for name in ("tune", "final", "all"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", choices=DATASETS, required=True)
        p.add_argument("--model", choices=MODEL_NAMES, required=name != "all")
        if name != "final":
            p.add_argument("--n-trials", type=int, default=None, help="total trials (default: 50 housing, 30 kc)")
            p.add_argument("--timeout", type=float, default=None, help="seconds")

    p = sub.add_parser("ablation")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))

    p = sub.add_parser("size-curve")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))

    p = sub.add_parser("summary")
    p.add_argument("--ml-results", type=Path, default=None, help="CSV of the ML team: dataset, model, test_rmse")

    sub.add_parser("replot", help="rebuild per-model figures from saved results (no training)")

    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows redirects stdout as cp1252 otherwise
    device = resolve_device(args.device)
    configure_torch(device, args.threads)

    if args.cmd == "splits":
        from src.splits import make_splits
        for ds in DATASETS:
            make_splits(ds, overwrite=args.overwrite)
    elif args.cmd in ("tune", "final", "all"):
        from src.final import run_final
        from src.tune import tune
        models = [args.model] if args.model else list(MODEL_NAMES)
        for model in models:
            if args.cmd in ("tune", "all"):
                tune(args.dataset, model, args.n_trials or DEFAULT_N_TRIALS[args.dataset], device, args.timeout)
            if args.cmd in ("final", "all"):
                run_final(args.dataset, model, device)
    elif args.cmd == "ablation":
        from experiments.ablation import run_ablation
        run_ablation(args.dataset, args.models, device)
    elif args.cmd == "size-curve":
        from experiments.size_curve import run_size_curve
        run_size_curve(args.models, device)
    elif args.cmd == "summary":
        from src.summary import build_summary
        build_summary(args.ml_results)
    elif args.cmd == "replot":
        from src.replot import replot_all
        replot_all()


if __name__ == "__main__":
    main()
