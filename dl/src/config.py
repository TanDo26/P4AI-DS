"""Paths, seeds and fixed experiment settings shared by every module."""
import os
from pathlib import Path

DL_DIR = Path(__file__).resolve().parents[1]
# The two Kaggle CSVs live one level above dl/ by default; override on Kaggle/Colab.
DATA_DIR = Path(os.environ.get("HOUSE_DATA_DIR", DL_DIR.parent))
SPLITS_DIR = DL_DIR / "splits"
RESULTS_DIR = DL_DIR / "results"
FIGURES_DIR = DL_DIR / "figures"

SEED_DATA = 42   # every split / resampling of data (train-test, KFold, inner_val, subsets, bootstrap)
SEED_MODEL = 36  # randomness inside models (team rule)

TEST_SIZE = 0.2
N_FOLDS = 5
INNER_VAL_SIZE = 0.15

DATASETS = ("housing", "kc")
CSV_FILES = {"housing": "Housing.csv", "kc": "kc_house_data.csv"}
MODEL_NAMES = ("ft_transformer", "tabnet")

DEFAULT_N_TRIALS = {"housing": 50, "kc": 30}

# Learning curve on King County: fold-train sizes (436 = size of the Housing train set).
SIZE_CURVE_SIZES = (436, 1000, 2500, 5000, 10000)

N_BOOTSTRAP = 1000
N_PERMUTATION_REPEATS = 5
