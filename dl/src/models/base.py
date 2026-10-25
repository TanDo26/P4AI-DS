"""Common interface so cv.py / tune.py / final.py treat every model the same way."""
import numpy as np


class BaseModel:
    kind: str  # which TabularPreprocessor output the model consumes

    best_epoch_: int | None = None  # epochs needed to reach the best inner_val score (1-based)
    history_: dict                  # {"train_loss": [...], "val_rmse": [...]} per epoch, standardized target scale

    def fit(self, X_tr, y_tr: np.ndarray, X_val=None, y_val: np.ndarray | None = None,
            n_epochs: int | None = None) -> "BaseModel":
        """With X_val: early stopping on it. Without X_val: train exactly n_epochs (final model)."""
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        """Predictions on the transformed (standardized log) target scale."""
        raise NotImplementedError
