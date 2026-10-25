"""Stateful preprocessing. Fit only on the rows whose labels train the weights (inner_train
during CV, the full train set for the final model), then transform everything else."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler

from .config import SEED_DATA
from .data import FeatureSpec

KINDS = ("numcat", "tabnet")
LOG_PRICE_MAX = 23.0  # exp(23) ~ 1e10: guards expm1 against overflow from a diverged model


class TabularPreprocessor:
    """kind="numcat": (x_num float32, x_cat int64 | None) for FT-Transformer.
    kind="tabnet": one float32 matrix [numeric | categorical index] plus cat_idxs / cat_dims."""

    def __init__(self, spec: FeatureSpec, kind: str, scaler: str = "quantile"):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if scaler not in ("quantile", "standard"):
            raise ValueError("scaler must be 'quantile' or 'standard'")
        self.spec, self.kind, self.scaler = spec, kind, scaler

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        s = self.spec
        if s.cont:
            if self.scaler == "quantile":
                self.cont_scaler_ = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(X)),
                                                        random_state=SEED_DATA)
            else:
                self.cont_scaler_ = StandardScaler()
            self.cont_scaler_.fit(X[s.cont].to_numpy(dtype=np.float64))
        if s.ord:
            self.ord_scaler_ = StandardScaler().fit(X[s.ord].to_numpy(dtype=np.float64))
        if s.cat:
            self.cat_encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.cat_encoder_.fit(X[s.cat].astype(str))
            # index 0 is reserved for categories never seen during fit
            self.cat_cardinalities_ = [len(c) + 1 for c in self.cat_encoder_.categories_]
        else:
            self.cat_cardinalities_ = []
        return self

    def transform_num(self, X: pd.DataFrame) -> np.ndarray:
        s, parts = self.spec, []
        if s.cont:
            parts.append(self.cont_scaler_.transform(X[s.cont].to_numpy(dtype=np.float64)))
        if s.ord:
            parts.append(self.ord_scaler_.transform(X[s.ord].to_numpy(dtype=np.float64)))
        if s.bin:
            parts.append(X[s.bin].to_numpy(dtype=np.float64))
        return np.hstack(parts).astype(np.float32)

    def transform_cat(self, X: pd.DataFrame) -> np.ndarray | None:
        if not self.spec.cat:
            return None
        return (self.cat_encoder_.transform(X[self.spec.cat].astype(str)) + 1).astype(np.int64)

    def transform(self, X: pd.DataFrame):
        x_num, x_cat = self.transform_num(X), self.transform_cat(X)
        if self.kind == "numcat":
            return x_num, x_cat
        return x_num if x_cat is None else np.hstack([x_num, x_cat.astype(np.float32)])

    @property
    def n_num(self) -> int:
        return len(self.spec.num)

    @property
    def feature_names(self) -> list[str]:
        """Column order of the transformed matrix (and token order of FT-Transformer after [CLS])."""
        return self.spec.num + self.spec.cat

    @property
    def cat_idxs(self) -> list[int]:
        return list(range(self.n_num, self.n_num + len(self.spec.cat)))


class TargetTransformer:
    """y' = (log1p(price) - mu) / sigma; inverse() maps predictions back to price."""

    def __init__(self, log: bool = True):
        self.log = log

    def _forward(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y) if self.log else np.asarray(y, dtype=np.float64)

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        z = self._forward(y)
        self.mean_, self.std_ = float(z.mean()), float(z.std())
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((self._forward(y) - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, t: np.ndarray) -> np.ndarray:
        z = np.asarray(t, dtype=np.float64).ravel() * self.std_ + self.mean_
        return np.expm1(np.minimum(z, LOG_PRICE_MAX)) if self.log else z


def preprocessing_markdown(dataset: str, model_name: str, spec: FeatureSpec, scaler: str, target_log: bool = True) -> str:
    """Human-readable record of every preprocessing step, saved next to the results."""
    stateless = {
        "housing": [
            "6 cột yes/no → 0/1",
            "`area` → `log(area)`",
            "`bedrooms`, `bathrooms`, `stories`, `parking` giữ dạng số",
            "`furnishingstatus` là categorical",
        ],
        "kc": [
            "Bỏ `id`; từ `date` tạo `days_since_start` (số ngày từ 02/05/2014) và `sale_month`",
            "Sửa `bedrooms = 33 → 3` (lỗi nhập liệu)",
            "`house_age = năm bán − yr_built` (clip ≥ 0); `years_since_update = năm bán − max(yr_built, yr_renovated)` (clip ≥ 0)",
            "`is_renovated`, `has_basement` (cờ 0/1); `dist_downtown_km` (khoảng cách tới trung tâm Seattle)",
            "`log1p` cho `sqft_living`, `sqft_lot`, `sqft_basement`, `sqft_living15`, `sqft_lot15`",
            "Bỏ `sqft_above` (= sqft_living − sqft_basement), `yr_built`, `yr_renovated`",
            "`zipcode` là categorical",
        ],
    }[dataset]
    scaler_name = "QuantileTransformer(output_distribution='normal')" if scaler == "quantile" else "StandardScaler"
    cat_note = ("index nguyên → embedding trong FT-Transformer" if model_name == "ft_transformer"
                else "index nguyên → embedding của TabNet (`cat_idxs`, `cat_dims`)")
    lines = [
        f"# Preprocessing — {dataset} / {model_name}",
        "",
        "## Bước stateless (từng dòng, không học tham số)",
        *[f"- {s}" for s in stateless],
        "",
        "## Bước stateful (fit trên inner_train trong CV; trên toàn bộ train cho model cuối)",
        "",
        "| Bước | Cột | Fit trên | Lý do |",
        "|---|---|---|---|",
        f"| {scaler_name} | {', '.join(spec.cont)} | inner_train / train | Cột liên tục, lệch phải; scaler chọn bằng CV |",
        f"| StandardScaler | {', '.join(spec.ord)} | inner_train / train | Cột thứ tự, ít giá trị |",
        f"| Giữ nguyên 0/1 | {', '.join(spec.bin)} | — | Cờ nhị phân |",
        f"| OrdinalEncoder (+1, index 0 = giá trị lạ) | {', '.join(spec.cat)} | inner_train / train | {cat_note} |",
        ("| Target: `log1p(price)` rồi chuẩn hoá (μ, σ) | price | inner_train / train | Giá lệch phải; dự đoán được đổi về giá gốc bằng `expm1` trước khi tính metric |"
         if target_log else
         "| Target: chuẩn hoá (μ, σ) trên giá gốc | price | inner_train / train | Ablation không dùng log |"),
        "",
    ]
    return "\n".join(lines)
