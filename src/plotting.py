"""
plotting.py — Helpers de visualização reutilizáveis.
Todos os plots retornam Figure para que io_utils.save_fig possa salvá-los.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (10, 8)
FIGSIZE_SQ = (8, 8)


# ── Genérico ──────────────────────────────────────────────────────────────────

def bar_with_proportion(
    series: pd.Series,
    title: str,
    xlabel: str = "",
    ylabel: str = "Contagem",
    color: str = "steelblue",
) -> plt.Figure:
    counts = series.value_counts()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor="white")
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{val:,}\n({val/total:.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def heatmap(
    df: pd.DataFrame,
    title: str,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
    annot: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def numeric_histograms(df: pd.DataFrame, cols: list[str], title: str) -> plt.Figure:
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, cols):
        df[col].dropna().plot.hist(ax=ax, bins=30, color="steelblue", edgecolor="white", density=True)
        df[col].dropna().plot.kde(ax=ax, color="crimson", linewidth=2)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def categorical_bars(df: pd.DataFrame, cols: list[str], title: str) -> plt.Figure:
    n = len(cols)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, cols):
        vc = df[col].value_counts()
        ax.barh(vc.index.astype(str), vc.values, color="mediumseagreen")
        ax.set_title(col, fontsize=10)
        ax.invert_yaxis()
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def boxplots_by_group(
    df: pd.DataFrame,
    cols: list[str],
    group_col: str,
    title: str,
) -> plt.Figure:
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, cols):
        df.boxplot(column=col, by=group_col, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(group_col)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def missingness_heatmap(df: pd.DataFrame) -> plt.Figure:
    missing = df.isnull()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(missing.T, cbar=False, cmap="viridis", ax=ax, yticklabels=True)
    ax.set_title("Heatmap de Valores Ausentes", fontsize=13, fontweight="bold")
    ax.set_xlabel("Índice da Instância")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


# ── Baseline ──────────────────────────────────────────────────────────────────

def confusion_matrix_plot(cm: np.ndarray, classes: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Real")
    ax.set_xlabel("Previsto")
    fig.tight_layout()
    return fig


def roc_curve_plot(fpr, tpr, auc_score: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curva ROC — Modelo Baseline", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def precision_recall_curve_plot(precision, recall, avg_precision: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {avg_precision:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_title("Curva Precisão-Recall — Modelo Baseline", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    df = importance_df.nlargest(top_n, "importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_title(f"Top-{top_n} Features por Importância (MDI)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância Média (MDI)")
    fig.tight_layout()
    return fig


def score_distribution(y_score: np.ndarray, y_true: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, name in [(0, "Não inadimplente"), (1, "Inadimplente")]:
        ax.hist(y_score[y_true == label], bins=40, alpha=0.6, label=name, density=True)
    ax.set_xlabel("Score de Probabilidade (classe 1)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição das Predições por Classe Real", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


# ── LIME ──────────────────────────────────────────────────────────────────────

def lime_contributions_bar(
    contributions: list[tuple[str, float]],
    title: str = "Contribuições LIME",
) -> plt.Figure:
    features, values = zip(*contributions)
    colors = ["#e05c5c" if v < 0 else "#4b9cd3" for v in values]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Contribuição LIME")
    fig.tight_layout()
    return fig


def feature_frequency_bar(freq_df: pd.DataFrame, title: str = "Frequência de Features nas Top-5") -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(freq_df["feature"], freq_df["count"], color="mediumseagreen")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Frequência")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ── Q1.1 ─────────────────────────────────────────────────────────────────────

def line_mean_std(
    x: list, y_mean: list, y_std: list,
    xlabel: str, ylabel: str, title: str,
    color: str = "steelblue",
) -> plt.Figure:
    x = np.array(x)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, y_mean, color=color, marker="o", linewidth=2, label="Média")
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color, label="±1 DP")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def overlap_heatmap(matrix: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.3, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("n_perturbations")
    ax.set_ylabel("Instância")
    fig.tight_layout()
    return fig


def boxplot_series(data: pd.Series, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data.dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue"))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def violin_by_n(df: pd.DataFrame, n_col: str, value_col: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    order = sorted(df[n_col].unique())
    sns.violinplot(data=df, x=n_col, y=value_col, order=order,
                   palette="muted", ax=ax, cut=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("n_perturbations")
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig


def histogram_adaptive_n(series: pd.Series, bins: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=bins, color="darkorange", edgecolor="white")
    ax.axvline(series.median(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mediana = {series.median():.0f}")
    ax.set_title("Distribuição do n Adaptativo por Instância", fontsize=13, fontweight="bold")
    ax.set_xlabel("n_perturbations adaptativo")
    ax.set_ylabel("Frequência")
    ax.legend()
    fig.tight_layout()
    return fig


def scatter_two_metrics(
    x: pd.Series, y: pd.Series,
    xlabel: str, ylabel: str, title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(x, y, alpha=0.6, color="steelblue", edgecolors="white", s=60)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def cost_vs_n(n_vals: list[int], times_mean: list[float], times_std: list[float]) -> plt.Figure:
    return line_mean_std(
        n_vals, times_mean, times_std,
        xlabel="n_perturbations",
        ylabel="Tempo médio (s)",
        title="Custo Computacional vs. n_perturbations",
        color="darkorange",
    )


def metric_correlation_heatmap(df: pd.DataFrame, title: str) -> plt.Figure:
    corr = df.select_dtypes(include="number").corr()
    return heatmap(corr, title, fmt=".2f", cmap="coolwarm")
