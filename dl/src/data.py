"""Loading and stateless feature engineering.

Everything here works row by row and learns nothing from the data, so it is safe to
run before splitting. Anything that learns statistics lives in preprocess.py.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CSV_FILES, DATA_DIR


@dataclass(frozen=True)
class FeatureSpec:
    cont: list[str]  # continuous -> QuantileTransformer or StandardScaler (chosen by CV)
    ord: list[str]   # small ordered integers -> StandardScaler
    bin: list[str]   # 0/1 flags -> passthrough
    cat: list[str]   # nominal -> integer index, embedded inside the model

    @property
    def num(self) -> list[str]:
        return self.cont + self.ord + self.bin

    @property
    def all(self) -> list[str]:
        return self.num + self.cat


HOUSING_YES_NO = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
HOUSING_ORD = ["bedrooms", "bathrooms", "stories", "parking"]

KC_SQFT = ["sqft_living", "sqft_lot", "sqft_basement", "sqft_living15", "sqft_lot15"]
KC_ORD = ["bedrooms", "bathrooms", "floors", "view", "condition", "grade"]
SEATTLE_DOWNTOWN = (47.6062, -122.3321)
KC_FIRST_SALE = pd.Timestamp("2014-05-02")


def load_raw(name: str) -> pd.DataFrame:
    path = DATA_DIR / CSV_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the Kaggle CSV there or set HOUSE_DATA_DIR.")
    return pd.read_csv(path)


def make_features_housing(df: pd.DataFrame, use_fe: bool = True) -> tuple[pd.DataFrame, FeatureSpec]:
    X = pd.DataFrame(index=df.index)
    X["area"] = np.log(df["area"]) if use_fe else df["area"].astype(float)
    for c in HOUSING_ORD:
        X[c] = df[c].astype(float)
    for c in HOUSING_YES_NO:
        X[c] = (df[c] == "yes").astype(float)
    X["furnishingstatus"] = df["furnishingstatus"].astype(str)
    return X, FeatureSpec(cont=["area"], ord=HOUSING_ORD, bin=HOUSING_YES_NO, cat=["furnishingstatus"])


def make_features_kc(df: pd.DataFrame, use_fe: bool = True) -> tuple[pd.DataFrame, FeatureSpec]:
    bedrooms = df["bedrooms"].replace(33, 3)  # data-entry typo: 1,620 sqft and 1.75 bathrooms
    zipcode = df["zipcode"].astype(str)

    if not use_fe:  # ablation: original columns, only id/date dropped
        cont = ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "sqft_living15", "sqft_lot15",
                "lat", "long", "yr_built", "yr_renovated"]
        X = df[cont + KC_ORD + ["waterfront"]].astype(float)
        X["bedrooms"] = bedrooms.astype(float)
        X["zipcode"] = zipcode
        return X, FeatureSpec(cont=cont, ord=KC_ORD, bin=["waterfront"], cat=["zipcode"])

    date = pd.to_datetime(df["date"].str[:8], format="%Y%m%d")
    sale_year = date.dt.year
    last_update = np.where(df["yr_renovated"] > 0, df["yr_renovated"], df["yr_built"])

    X = pd.DataFrame(index=df.index)
    for c in KC_SQFT:
        X[c] = np.log1p(df[c])
    X["lat"] = df["lat"]
    X["long"] = df["long"]
    lat0, lon0 = SEATTLE_DOWNTOWN
    X["dist_downtown_km"] = 111.0 * np.hypot(df["lat"] - lat0, (df["long"] - lon0) * np.cos(np.radians(lat0)))
    X["days_since_start"] = (date - KC_FIRST_SALE).dt.days.astype(float)
    X["house_age"] = (sale_year - df["yr_built"]).clip(lower=0).astype(float)
    X["years_since_update"] = (sale_year - last_update).clip(lower=0).astype(float)
    for c in KC_ORD:
        X[c] = df[c].astype(float)
    X["bedrooms"] = bedrooms.astype(float)
    X["sale_month"] = date.dt.month.astype(float)
    X["waterfront"] = df["waterfront"].astype(float)
    X["is_renovated"] = (df["yr_renovated"] > 0).astype(float)
    X["has_basement"] = (df["sqft_basement"] > 0).astype(float)
    X["zipcode"] = zipcode

    cont = KC_SQFT + ["lat", "long", "dist_downtown_km", "days_since_start", "house_age", "years_since_update"]
    return X, FeatureSpec(cont=cont, ord=KC_ORD + ["sale_month"],
                          bin=["waterfront", "is_renovated", "has_basement"], cat=["zipcode"])


def load_dataset(name: str, use_fe: bool = True) -> tuple[pd.DataFrame, np.ndarray, FeatureSpec]:
    """Return features (row i == CSV row i), price in original units, and the column spec."""
    df = load_raw(name)
    make = {"housing": make_features_housing, "kc": make_features_kc}[name]
    X, spec = make(df, use_fe=use_fe)
    return X[spec.all], df["price"].to_numpy(dtype=np.float64), spec
