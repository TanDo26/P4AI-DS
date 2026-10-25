"""Model explanation: permutation importance (any model) plus model-specific views.
These functions receive data from final.py and never load the test split themselves."""
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import N_PERMUTATION_REPEATS, SEED_DATA
from .metrics import rmse


def permutation_importance(predict_price: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame,
                           y_true: np.ndarray, n_repeats: int = N_PERMUTATION_REPEATS,
                           seed: int = SEED_DATA) -> pd.DataFrame:
    """Increase of RMSE (original price) when one feature column is shuffled."""
    rng = np.random.default_rng(seed)
    base = rmse(y_true, predict_price(X))
    rows = []
    for col in X.columns:
        increases = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            increases.append(rmse(y_true, predict_price(Xp)) - base)
        rows.append({"feature": col, "rmse_increase_mean": float(np.mean(increases)),
                     "rmse_increase_std": float(np.std(increases, ddof=1))})
    return pd.DataFrame(rows).sort_values("rmse_increase_mean", ascending=False, ignore_index=True)


@torch.no_grad()
def ft_attention_importance(model, X, feature_names: list[str], batch_size: int = 2048) -> pd.DataFrame:
    """Average attention from the [CLS] query to every feature token, over heads, blocks and rows.

    rtdl_revisiting_models does not return attention maps, so a forward pre-hook grabs each attention
    module's inputs and recomputes softmax(Q K^T / sqrt(d_head)) for the [CLS] query.
    """
    net = model.net_
    net.eval()
    sums = torch.zeros(len(feature_names) + 1, dtype=torch.float64)
    count = 0

    def hook(module, args):
        nonlocal sums, count
        x_q, x_kv = args
        q, k = module.W_q(x_q[:, :1]), module.W_k(x_kv)
        n_heads = module._n_heads
        b, n_tokens, d = k.shape
        q = q.reshape(b, 1, n_heads, d // n_heads).transpose(1, 2)
        k = k.reshape(b, n_tokens, n_heads, d // n_heads).transpose(1, 2)
        probs = torch.softmax(q @ k.transpose(-1, -2) / (d // n_heads) ** 0.5, dim=-1)  # (b, heads, 1, tokens)
        sums += probs.mean(dim=1).squeeze(1).sum(dim=0).double().cpu()
        count += b

    handles = [block["attention"].register_forward_pre_hook(hook) for block in net.backbone.blocks]
    try:
        xn, xc = model._tensors(X)
        for i in range(0, len(xn), batch_size):
            net(xn[i:i + batch_size], None if xc is None else xc[i:i + batch_size])
    finally:
        for h in handles:
            h.remove()
    att = (sums / count).numpy()[1:]  # token 0 is [CLS]; then numeric features, then categorical
    return (pd.DataFrame({"feature": feature_names, "attention": att / att.sum()})
            .sort_values("attention", ascending=False, ignore_index=True))


def tabnet_importance(model, X_matrix: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Global importance from training (feature_importances_) and from test rows (explain), plus masks."""
    explain_matrix, masks = model.explain(X_matrix)
    test_imp = explain_matrix.sum(axis=0)
    df = pd.DataFrame({"feature": feature_names,
                       "importance_train": model.reg_.feature_importances_,
                       "importance_test": test_imp / test_imp.sum()})
    return df.sort_values("importance_test", ascending=False, ignore_index=True), masks
