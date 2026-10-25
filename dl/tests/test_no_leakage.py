"""Guards for the comparison rules: shared split, folds inside train, preprocessing fit on
inner_train only, and the test split read by final.py alone."""
import re

import numpy as np
import pytest
from sklearn.model_selection import KFold, train_test_split

from src.config import DATASETS, DL_DIR, N_FOLDS, SEED_DATA, TEST_SIZE
from src.data import load_dataset, load_raw
from src.preprocess import TabularPreprocessor, TargetTransformer
from src.splits import (fold_indices, inner_split, load_folds, load_test_idx, load_train_idx, make_splits,
                        subset_rows)


@pytest.fixture(scope="module", autouse=True)
def splits():
    for ds in DATASETS:
        make_splits(ds)  # no-op when the shared files already exist


@pytest.mark.parametrize("ds", DATASETS)
def test_train_test_disjoint_and_complete(ds):
    train, test = load_train_idx(ds), load_test_idx(ds)
    n = len(load_raw(ds))
    assert not set(train) & set(test)
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(n))
    assert len(test) == int(np.ceil(TEST_SIZE * n))


@pytest.mark.parametrize("ds", DATASETS)
def test_split_matches_plain_sklearn_call(ds):
    """The ML team reproduces the same split and folds with the usual sklearn calls on the raw CSV."""
    df = load_raw(ds)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    assert np.array_equal(train_df.index.to_numpy(), load_train_idx(ds))
    assert np.array_equal(test_df.index.to_numpy(), load_test_idx(ds))
    folds = load_folds(ds)
    for k, (_, val_pos) in enumerate(KFold(5, shuffle=True, random_state=42).split(train_df)):
        assert set(train_df.index[val_pos]) == set(folds.loc[folds["fold"] == k, "row"])


@pytest.mark.parametrize("ds", DATASETS)
def test_folds_partition_train(ds):
    folds, train = load_folds(ds), load_train_idx(ds)
    assert sorted(folds["row"]) == sorted(train)
    sizes = folds["fold"].value_counts()
    assert len(sizes) == N_FOLDS and sizes.max() - sizes.min() <= 1


@pytest.mark.parametrize("ds", DATASETS)
def test_inner_split_stays_inside_fold_train(ds):
    folds, test = load_folds(ds), set(load_test_idx(ds))
    for k in range(N_FOLDS):
        fit_rows, val_rows = fold_indices(folds, k)
        inner_tr, inner_val = inner_split(fit_rows)
        assert not set(inner_tr) & set(inner_val)
        assert set(inner_tr) | set(inner_val) == set(fit_rows)
        assert not (set(fit_rows) & set(val_rows)) and not (set(fit_rows) & test)


def test_preprocessing_statistics_come_from_inner_train_only():
    X, y, spec = load_dataset("kc")
    fit_rows, _ = fold_indices(load_folds("kc"), 0)
    inner_tr, _ = inner_split(fit_rows)
    pre = TabularPreprocessor(spec, "numcat", scaler="standard").fit(X.iloc[inner_tr])
    assert np.allclose(pre.cont_scaler_.mean_, X.iloc[inner_tr][spec.cont].mean().to_numpy())
    assert np.allclose(pre.ord_scaler_.mean_, X.iloc[inner_tr][spec.ord].mean().to_numpy())
    tt = TargetTransformer().fit(y[inner_tr])
    assert np.isclose(tt.mean_, np.log1p(y[inner_tr]).mean())


def test_only_final_reads_test_split():
    callers = set()
    for path in [*(DL_DIR / "src").rglob("*.py"), *(DL_DIR / "experiments").rglob("*.py"), DL_DIR / "run.py"]:
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?<!def )load_test_idx\(", text):
            callers.add(path.name)
    assert callers == {"final.py"}


def test_housing_csv_is_sorted_so_folds_must_be_shuffled():
    df = load_raw("housing")
    assert df["price"].is_monotonic_decreasing  # why every split uses shuffle=True
    folds = load_folds("housing")
    prices = df["price"].to_numpy()
    overall = prices[folds["row"]].mean()
    for k in range(N_FOLDS):
        assert abs(prices[folds.loc[folds["fold"] == k, "row"]].mean() / overall - 1) < 0.2


def test_size_subsets_nested_and_inside_fold_train():
    folds, test = load_folds("kc"), set(load_test_idx("kc"))
    for k in range(N_FOLDS):
        fit_rows, val_rows = fold_indices(folds, k)
        small, large = set(subset_rows("kc", k, 436)), set(subset_rows("kc", k, 1000))
        assert len(small) == 436 and small <= large <= set(fit_rows)
        assert not large & (set(val_rows) | test)


def test_seed_constants():
    assert SEED_DATA == 42
