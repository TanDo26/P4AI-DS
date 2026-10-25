"""Train/test split, 5 CV folds and learning-curve subsets, saved once and shared with the ML team.

Row ids are 0-based positions in the original CSV. train_idx.csv keeps the order returned by
train_test_split, so KFold(5, shuffle=True, random_state=42) on X_train reproduces folds.csv.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .config import INNER_VAL_SIZE, N_FOLDS, SEED_DATA, SPLITS_DIR, TEST_SIZE
from .data import load_raw


def make_splits(name: str, overwrite: bool = False) -> None:
    out = SPLITS_DIR / name
    if (out / "folds.csv").exists() and not overwrite:
        print(f"[splits] {name}: already exists, keeping it (use --overwrite to regenerate)")
        return
    out.mkdir(parents=True, exist_ok=True)

    n = len(load_raw(name))
    train_idx, test_idx = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=SEED_DATA, shuffle=True)
    folds = np.empty(len(train_idx), dtype=int)
    for k, (_, val_pos) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED_DATA).split(train_idx)):
        folds[val_pos] = k

    pd.DataFrame({"row": train_idx}).to_csv(out / "train_idx.csv", index=False)
    pd.DataFrame({"row": test_idx}).to_csv(out / "test_idx.csv", index=False)
    pd.DataFrame({"row": train_idx, "fold": folds}).to_csv(out / "folds.csv", index=False)

    # Nested subsets for the learning curve: subset of size n in fold k = rows with rank < n.
    rng = np.random.default_rng(SEED_DATA)
    parts = []
    for k in range(N_FOLDS):
        fit_rows = train_idx[folds != k]
        parts.append(pd.DataFrame({"fold": k, "row": rng.permutation(fit_rows), "rank": np.arange(len(fit_rows))}))
    pd.concat(parts).to_csv(out / "size_subsets.csv", index=False)
    print(f"[splits] {name}: train={len(train_idx)} test={len(test_idx)} -> {out}")


def _read(name: str, file: str) -> pd.DataFrame:
    path = SPLITS_DIR / name / file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python run.py splits` first")
    return pd.read_csv(path)


def load_train_idx(name: str) -> np.ndarray:
    return _read(name, "train_idx.csv")["row"].to_numpy()


def load_test_idx(name: str) -> np.ndarray:
    """Test rows. Only final.py may call this (checked by tests/test_no_leakage.py)."""
    return _read(name, "test_idx.csv")["row"].to_numpy()


def load_folds(name: str) -> pd.DataFrame:
    return _read(name, "folds.csv")


def fold_indices(folds: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(fold_train rows, fold_val rows) for fold k."""
    return folds.loc[folds["fold"] != k, "row"].to_numpy(), folds.loc[folds["fold"] == k, "row"].to_numpy()


def inner_split(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split fold_train into inner_train (fit weights) and inner_val (early stopping only)."""
    inner_train, inner_val = train_test_split(rows, test_size=INNER_VAL_SIZE, random_state=SEED_DATA, shuffle=True)
    return inner_train, inner_val


def subset_rows(name: str, k: int, n: int) -> np.ndarray:
    """First n rows of fold k's fixed random permutation of fold_train (nested across n)."""
    subsets = _read(name, "size_subsets.csv")
    part = subsets[subsets["fold"] == k]
    return part.loc[part["rank"] < n, "row"].to_numpy()
