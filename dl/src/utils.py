"""Small helpers: seeding, device, timing, JSON and output folders."""
import json
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from .config import FIGURES_DIR, RESULTS_DIR, SEED_MODEL


def set_seed(seed: int = SEED_MODEL) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def configure_torch(device: str, threads: int | None = None) -> None:
    if threads:
        torch.set_num_threads(threads)
    if device == "cpu":
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def _to_builtin(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_to_builtin), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def results_dir(dataset: str, model_name: str) -> Path:
    path = RESULTS_DIR / dataset / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_figures_dir(dataset: str, model_name: str) -> Path:
    path = results_dir(dataset, model_name) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(name: str) -> Path:
    path = FIGURES_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def env_info(device: str) -> dict:
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else (platform.processor() or platform.machine())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "hardware": hardware,
        "torch_threads": torch.get_num_threads(),
        "packages": {p: _pkg_version(p) for p in
                     ["torch", "numpy", "pandas", "scikit-learn", "optuna", "pytorch-tabnet", "rtdl_revisiting_models"]},
    }
