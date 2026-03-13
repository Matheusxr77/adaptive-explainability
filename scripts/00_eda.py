"""
00_eda.py — Análise Exploratória do dataset de risco de crédito.
Persiste CSVs e gráficos em resultados/.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Insere a raiz no path para importações locais
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_PATH, TARGET_COL, NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS,
    EDA_MISSINGNESS_CSV, EDA_NUMERIC_CSV, EDA_CATEGORICAL_CSV, EDA_INSIGHTS_CSV,
    FIG_DIR,
)
from src.io_utils import save_csv, save_fig
import src.plotting as P

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_eda() -> pd.DataFrame:
    logger.info("=== EDA | Início ===")

    # ── Carregamento ──────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    logger.info("Dataset carregado: %d linhas × %d colunas", *df.shape)

    expected_cols = NUMERICAL_COLS + CATEGORICAL_COLS + ORDINAL_COLS + [TARGET_COL]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes no dataset: {missing_cols}")

    # ── Remoção de outliers ───────────────────────────────────────────────────
    n_before = len(df)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)]
    logger.info("Outliers removidos: %d → %d linhas", n_before, len(df))

    all_feats = NUMERICAL_COLS + CATEGORICAL_COLS + ORDINAL_COLS

    # ── 1. Valores Ausentes ───────────────────────────────────────────────────
    missing = df[all_feats].isnull().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    save_csv(missing, EDA_MISSINGNESS_CSV)

    fig = P.missingness_heatmap(df[all_feats].sample(min(3000, len(df)), random_state=42))
    save_fig(fig, FIG_DIR / "eda_missing_heatmap.png")

    # ── 2. Distribuição do Target ─────────────────────────────────────────────
    fig = P.bar_with_proportion(df[TARGET_COL], "Distribuição do Target (loan_status)",
                                 xlabel="loan_status (0=não inadimplente, 1=inadimplente)")
    save_fig(fig, FIG_DIR / "eda_target_distribution.png")

    # ── 3. Sumário numérico ───────────────────────────────────────────────────
    num_summary = df[NUMERICAL_COLS].describe().T.reset_index().rename(columns={"index": "feature"})
    save_csv(num_summary, EDA_NUMERIC_CSV)

    fig = P.numeric_histograms(df, NUMERICAL_COLS, "Distribuições das Features Numéricas")
    save_fig(fig, FIG_DIR / "eda_numeric_histograms.png")

    # ── 4. Sumário categórico ─────────────────────────────────────────────────
    cat_summary_rows = []
    for col in CATEGORICAL_COLS + ORDINAL_COLS:
        vc = df[col].value_counts()
        for val, cnt in vc.items():
            cat_summary_rows.append({"feature": col, "value": val, "count": cnt,
                                     "pct": round(cnt / len(df) * 100, 2)})
    cat_summary = pd.DataFrame(cat_summary_rows)
    save_csv(cat_summary, EDA_CATEGORICAL_CSV)

    fig = P.categorical_bars(df, CATEGORICAL_COLS + ORDINAL_COLS, "Distribuição das Features Categóricas")
    save_fig(fig, FIG_DIR / "eda_categorical_bars.png")

    # ── 5. Correlação ─────────────────────────────────────────────────────────
    corr_df = df[NUMERICAL_COLS + [TARGET_COL]].corr()
    fig = P.heatmap(corr_df, "Matriz de Correlação (Features Numéricas)")
    save_fig(fig, FIG_DIR / "eda_correlation_heatmap.png")

    # ── 6. Boxplots por Target ─────────────────────────────────────────────────
    fig = P.boxplots_by_group(df, NUMERICAL_COLS, TARGET_COL, "Boxplots por loan_status")
    save_fig(fig, FIG_DIR / "eda_boxplots_by_target.png")

    # ── 7. Insights para relatório ────────────────────────────────────────────
    class1_pct = df[TARGET_COL].mean() * 100
    most_missing = missing.sort_values("missing_pct", ascending=False).iloc[0]
    top_corr = (
        corr_df[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .idxmax()
    )
    top_corr_val = corr_df.loc[top_corr, TARGET_COL]

    insights = pd.DataFrame([
        {"insight": "proporção_inadimplentes", "valor": f"{class1_pct:.1f}%"},
        {"insight": "feature_mais_ausente", "valor": most_missing["feature"],
         "detalhe": f"{most_missing['missing_pct']:.1f}% ausentes"},
        {"insight": "feature_mais_correlacionada_com_target",
         "valor": top_corr, "detalhe": f"corr={top_corr_val:.3f}"},
        {"insight": "tamanho_apos_limpeza", "valor": str(len(df))},
    ])
    save_csv(insights, EDA_INSIGHTS_CSV)

    logger.info("=== EDA | Concluído ===")
    return df


if __name__ == "__main__":
    run_eda()
