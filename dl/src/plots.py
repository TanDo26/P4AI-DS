"""Static report figures (matplotlib). Colors follow a fixed, CVD-validated categorical order:
a model keeps the same color in every figure; text and axes use neutral ink, never series color."""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # semibold fallback notices
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"ft_transformer": CATEGORICAL[0], "tabnet": CATEGORICAL[1]}
MODEL_LABELS = {"ft_transformer": "FT-Transformer", "tabnet": "TabNet"}
DATASET_LABELS = {"housing": "Housing (545 dòng)", "kc": "King County (21.613 dòng)"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.labelcolor": INK_2, "lines.linewidth": 2.0, "savefig.dpi": 160,
})


def model_color(name: str) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    extra = [c for c in CATEGORICAL if c not in MODEL_COLORS.values()]  # ML models joining later
    MODEL_COLORS[name] = extra[(len(MODEL_COLORS) - 2) % len(extra)]
    return MODEL_COLORS[name]


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _money(x, _pos=None) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.3g}{suffix}"
    return f"{x:.3g}"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def learning_curve(history: dict, best_epoch: int | None, path: Path, title: str) -> None:
    """Train vs inner_val error per epoch for one fold; both on the standardized log-price scale."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    train = np.sqrt(np.clip(history["train_loss"], 0, None))
    epochs = np.arange(1, len(train) + 1)
    ax.plot(epochs, train, color=CATEGORICAL[0], label="Train (√loss)")
    if history.get("val_rmse"):
        val = np.asarray(history["val_rmse"])
        ax.plot(np.arange(1, len(val) + 1), val, color=CATEGORICAL[1], label="inner_val RMSE")
        ax.annotate("inner_val", (len(val), val[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, va="center", fontsize=9)
    ax.annotate("train", (len(train), train[-1]), xytext=(4, 0), textcoords="offset points",
                color=INK_2, va="center", fontsize=9)
    if best_epoch:
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"best epoch = {best_epoch}", (best_epoch, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", color=INK_2, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (log-price chuẩn hoá)")
    ax.set_title(title, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _save(fig, path)


def pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", label="y = ŷ")
    ax.scatter(y_true, y_pred, s=14 if len(y_true) < 500 else 6, color=color, alpha=0.55,
               edgecolors=SURFACE, linewidths=0.4, label="Căn nhà (test)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(FuncFormatter(_money))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Giá thực tế")
    ax.set_ylabel("Giá dự đoán")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def residual_hist(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str, color: str) -> None:
    pct = 100 * (y_pred - y_true) / y_true
    lo, hi = np.percentile(pct, [1, 99])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(np.clip(pct, lo, hi), bins=40, color=color, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axvline(np.median(pct), color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"trung vị {np.median(pct):+.1f}%", (np.median(pct), ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlabel("Sai số tương đối (ŷ − y) / y  [%]  (cắt ở phân vị 1–99)")
    ax.set_ylabel("Số căn nhà")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def importance_bars(df: pd.DataFrame, value: str, path: Path, title: str, xlabel: str, color: str,
                    err: str | None = None, top: int = 15, money: bool = False) -> None:
    d = df.sort_values(value, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * len(d) + 1.2))
    ax.barh(d["feature"], d[value], color=color, height=0.7,
            xerr=None if err is None else d[err], error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    if money:
        ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    _save(fig, path)


def tabnet_masks(masks: dict, feature_names: list[str], path: Path, title: str, n_rows: int = 12) -> None:
    """Small multiples: one heatmap per decision step, rows = sample test houses."""
    steps = sorted(masks)
    vmax = max(float(masks[s][:n_rows].max()) for s in steps) or 1.0
    fig, axes = plt.subplots(1, len(steps), figsize=(1.0 + 0.28 * len(feature_names) * len(steps), 0.25 * n_rows + 2.2),
                             sharey=True, squeeze=False)
    for ax, s in zip(axes[0], steps):
        im = ax.imshow(masks[s][:n_rows], aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        ax.set_title(f"Bước {s + 1}", fontsize=10)
        ax.set_xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=7)
        ax.grid(False)
    axes[0][0].set_ylabel("Căn nhà (test)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Trọng số mask")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    _save(fig, path)


def optuna_history(trials: pd.DataFrame, path: Path, title: str, color: str) -> None:
    done = trials[trials["state"] == "COMPLETE"].sort_values("number")
    if done.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.scatter(done["number"], done["value"], s=18, color=color, alpha=0.5, edgecolors=SURFACE,
               linewidths=0.5, label="Trial hoàn thành")
    ax.plot(done["number"], done["value"].cummin(), color=INK_2, linewidth=2, drawstyle="steps-post",
            label="Tốt nhất đến hiện tại")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV RMSE (giá gốc)")
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def param_importance(importances: dict, path: Path, title: str, color: str) -> None:
    if not importances:
        return
    d = pd.DataFrame({"feature": list(importances), "importance": list(importances.values())})
    importance_bars(d, "importance", path, title, "Mức quan trọng (fANOVA)", color)


def cv_comparison(summary: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str, money: bool) -> None:
    """One panel per dataset (RMSE is not comparable across currencies), one bar per model, ±1 std."""
    datasets = [d for d in ("housing", "kc") if d in set(summary["dataset"])]
    n_max = int(summary.groupby("dataset").size().max())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = summary[summary["dataset"] == ds]
        x = np.arange(len(d))
        ax.bar(x, d[f"cv_{metric}_mean"], yerr=d[f"cv_{metric}_std"], width=0.6,
               color=[model_color(m) for m in d["model"]], edgecolor=SURFACE, linewidth=2,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        for xi, v, sd in zip(x, d[f"cv_{metric}_mean"], d[f"cv_{metric}_std"]):
            ax.annotate(_money(v) if money else f"{v:.3f}", (xi, v + sd), xytext=(0, 4), textcoords="offset points",
                        ha="center", color=INK_2, fontsize=9)
        ax.set_xlim(-0.8, n_max - 0.2)  # same bar width in every panel, even with fewer models
        ax.set_xticks(x, [model_label(m) for m in d["model"]])
        ax.set_title(DATASET_LABELS[ds])
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        if money:
            ax.yaxis.set_major_formatter(FuncFormatter(_money))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def time_vs_rmse(summary: pd.DataFrame, path: Path, title: str) -> None:
    datasets = [d for d in ("housing", "kc") if d in set(summary["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = summary[summary["dataset"] == ds]
        for _, r in d.iterrows():
            ax.scatter(r["fit_time"], r["test_rmse"], s=70, color=model_color(r["model"]),
                       edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.annotate(model_label(r["model"]), (r["fit_time"], r["test_rmse"]), xytext=(6, 4),
                        textcoords="offset points", color=INK_2, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Thời gian train model cuối (s, log)")
        ax.set_ylabel("Test RMSE (giá gốc)")
        ax.yaxis.set_major_formatter(FuncFormatter(_money))
        ax.set_title(DATASET_LABELS[ds])
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def size_curve(df: pd.DataFrame, path: Path, title: str) -> None:
    """RMSE and R2 vs number of training rows (two panels, never a dual axis)."""
    agg = df.groupby(["model", "n_fold_train"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                                                    r2=("r2", "mean"), r2_sd=("r2", "std")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for metric, ax, ylabel in (("rmse", axes[0], "CV RMSE (giá gốc)"), ("r2", axes[1], "CV R²")):
        for model, g in agg.groupby("model"):
            g = g.sort_values("n_fold_train")
            c = model_color(model)
            ax.plot(g["n_fold_train"], g[metric], color=c, marker="o", markersize=5, label=model_label(model))
            ax.fill_between(g["n_fold_train"], g[metric] - g[f"{metric}_sd"], g[metric] + g[f"{metric}_sd"],
                            color=c, alpha=0.15, linewidth=0)
        ax.axvline(436, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("= cỡ train Housing", (436, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                    color=INK_2, fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sorted(agg["n_fold_train"].unique())))
        ax.xaxis.set_major_formatter(FuncFormatter(_money))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Số dòng train mỗi fold (log)")
        ax.set_ylabel(ylabel)
        if metric == "rmse":
            ax.yaxis.set_major_formatter(FuncFormatter(_money))
    axes[1].legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def ablation(df: pd.DataFrame, path: Path, title: str) -> None:
    """Horizontal grouped bars: CV RMSE per preprocessing variant, one color per model."""
    variants = list(dict.fromkeys(df["variant"]))
    models = list(dict.fromkeys(df["model"]))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(variants) * len(models) + 1.2))
    y = np.arange(len(variants))
    for i, m in enumerate(models):
        d = df[df["model"] == m].set_index("variant").reindex(variants)
        ax.barh(y + i * h, d["rmse_mean"], height=h * 0.9, xerr=d["rmse_std"], color=model_color(m),
                label=model_label(m), error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y + h * (len(models) - 1) / 2, variants)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_xlabel("CV RMSE (giá gốc), ±1 std")
    ax.set_title(title, pad=24)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), borderaxespad=0.2)
    _save(fig, path)
