"""
04_relatorio_final.py — Gera relatorio/relatorio_final.md a partir dos CSVs e figuras.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    EDA_INSIGHTS_CSV, BASELINE_METRICS_CSV, BASELINE_IMPORTANCE_CSV,
    Q11_N_LEVEL_CSV, Q11_ADAPTIVE_CSV, Q11_CONCLUSIONS_CSV,
    LIME_BASE_TEXTS_CSV, Q11_TEXTS_CSV,
    FIG_DIR, REPORT_MD, REPORT_CONCLUSIONS_CSV, REPORT_METRICS_CSV,
    LIME_N_PERTURBATIONS_GRID, SPEARMAN_THRESHOLD, TOP_K,
)
from src.io_utils import save_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _fig_ref(name: str, caption: str) -> str:
    rel = Path("../resultados/figuras") / name
    return f"![{caption}]({rel})\n*{caption}*\n"


def _table_md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def safe_load(path: Path, default=None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        logger.warning("Não foi possível carregar %s: %s", path, e)
        return default


def run_report() -> None:
    logger.info("=== Relatório Final | Início ===")

    eda = safe_load(EDA_INSIGHTS_CSV)
    baseline = safe_load(BASELINE_METRICS_CSV)
    importance = safe_load(BASELINE_IMPORTANCE_CSV)
    n_level = safe_load(Q11_N_LEVEL_CSV)
    adaptive = safe_load(Q11_ADAPTIVE_CSV)
    conclusions = safe_load(Q11_CONCLUSIONS_CSV)
    base_texts = safe_load(LIME_BASE_TEXTS_CSV)
    q11_texts = safe_load(Q11_TEXTS_CSV)

    # ── Métricas para sumário ─────────────────────────────────────────────────
    has_baseline = baseline is not None and not baseline.empty
    has_q11 = n_level is not None and adaptive is not None
    has_conclusions = conclusions is not None and not conclusions.empty

    auc_val = float(baseline["auc_roc"].iloc[0]) if has_baseline else "N/A"
    f1_val = float(baseline["f1_macro"].iloc[0]) if has_baseline else "N/A"
    median_n = int(adaptive["adaptive_n"].median()) if has_q11 else "N/A"
    spearman_n25 = float(n_level[n_level["n_perturbations"] == 25]["spearman_mean"].iloc[0]) if has_q11 else "N/A"
    spearman_n1000 = float(n_level[n_level["n_perturbations"] == 1000]["spearman_mean"].iloc[0]) if has_q11 else "N/A"

    top3_feats = (
        importance.head(3)["feature"].tolist() if importance is not None
        else ["N/A", "N/A", "N/A"]
    )

    slm_available = False
    if base_texts is not None:
        slm_available = (base_texts["llm_status"] == "ok").any()

    # ── Salva CSVs de sumário ─────────────────────────────────────────────────
    metrics_summary = pd.DataFrame([
        {"etapa": "baseline", "metrica": "auc_roc", "valor": auc_val},
        {"etapa": "baseline", "metrica": "f1_macro", "valor": f1_val},
        {"etapa": "q1.1", "metrica": "n_adaptativo_mediana", "valor": median_n},
        {"etapa": "q1.1", "metrica": "spearman_com_n25", "valor": spearman_n25},
        {"etapa": "q1.1", "metrica": "spearman_com_n1000", "valor": spearman_n1000},
    ])
    save_csv(metrics_summary, REPORT_METRICS_CSV)

    conclusions_data = pd.DataFrame([
        {"questao": "Q1.1",
         "pergunta": "Qual o menor n_perturbations que produz explicações estáveis?",
         "resposta": f"n = {median_n} (mediana sobre as instâncias de teste)",
         "criterio": f"Spearman ≥ {SPEARMAN_THRESHOLD} entre n e 2n"},
    ])
    save_csv(conclusions_data, REPORT_CONCLUSIONS_CSV)

    # ── Geração do Markdown ───────────────────────────────────────────────────
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = []

    lines += [
        "# Relatório Final — Explicabilidade Adaptativa via SLMs Locais",
        "",
        f"> **Gerado automaticamente em:** {now}  ",
        "> **Questão de pesquisa:** Q1.1 — Adaptatividade do número de perturbações no LIME  ",
        "> **Dataset:** credit_risk_dataset.csv (classificação de risco de crédito)  ",
        "> **Modelo:** RandomForestClassifier | **SLM:** ai/qwen2.5 (Docker Model Runner)  ",
        "",
        "---",
        "",
    ]

    # ── 1. Introdução ─────────────────────────────────────────────────────────
    lines += [
        "## 1. Introdução",
        "",
        "Este experimento aplica a metodologia de Zeng & Zhu (2024) para avaliar a "
        "interpretabilidade de um modelo de classificação de risco de crédito.",
        "",
        "O pipeline segue quatro etapas:",
        "1. **Input Representation** — Valores LIME estruturados como tuplas (feature, valor)",
        "2. **Prompt Engineering** — Construção de prompt contextualizado para o SLM",
        "3. **Geração local pelo SLM** — Resposta em linguagem natural via Docker Model Runner",
        "4. **Avaliação** — Métricas de estabilidade, sobreposição e coerência textual",
        "",
        f"O SLM estava {'disponível' if slm_available else 'indisponível (métricas textuais puladas)'}",
        "durante a execução deste pipeline.",
        "",
    ]

    # ── 2. Análise Exploratória ───────────────────────────────────────────────
    lines += [
        "## 2. Análise Exploratória do Dataset",
        "",
        _fig_ref("eda_target_distribution.png", "Distribuição do Target (loan_status)"),
        "",
        _fig_ref("eda_missing_heatmap.png", "Heatmap de Valores Ausentes"),
        "",
        _fig_ref("eda_numeric_histograms.png", "Distribuições das Features Numéricas"),
        "",
        _fig_ref("eda_categorical_bars.png", "Frequências das Features Categóricas"),
        "",
        _fig_ref("eda_correlation_heatmap.png", "Matriz de Correlação"),
        "",
        _fig_ref("eda_boxplots_by_target.png", "Boxplots por Classe Target"),
        "",
    ]

    if eda is not None:
        lines += ["**Insights principais:**", ""]
        for _, row in eda.iterrows():
            detalhe = row.get("detalhe", "")
            suffix = f" — {detalhe}" if pd.notna(detalhe) and detalhe else ""
            lines.append(f"- **{row['insight']}**: {row['valor']}{suffix}")
        lines.append("")

    # ── 3. Modelo Baseline ────────────────────────────────────────────────────
    lines += [
        "## 3. Modelo Baseline",
        "",
    ]

    if has_baseline:
        lines += [
            "| Métrica | Valor |",
            "|---|---|",
            f"| AUC-ROC | **{auc_val:.4f}** |",
            f"| F1-macro | {f1_val:.4f} |",
            f"| Accuracy | {float(baseline['accuracy'].iloc[0]):.4f} |",
            "",
        ]

    lines += [
        _fig_ref("baseline_roc_curve.png", "Curva ROC do Modelo Baseline"),
        "",
        _fig_ref("baseline_confusion_matrix.png", "Matriz de Confusão"),
        "",
        _fig_ref("baseline_precision_recall_curve.png", "Curva Precisão-Recall"),
        "",
        _fig_ref("baseline_feature_importance_top15.png", "Top-15 Features por Importância (MDI)"),
        "",
        _fig_ref("baseline_prediction_score_distribution.png", "Distribuição dos Scores por Classe"),
        "",
        f"As features mais importantes globalmente são: **{', '.join(top3_feats)}**.",
        "",
    ]

    # ── 4. Resultados Q1.1 ────────────────────────────────────────────────────
    lines += [
        "## 4. Resultados — Q1.1: Adaptatividade do LIME",
        "",
        "### 4.1 Estabilidade Numérica",
        "",
        _fig_ref("q11_spearman_vs_n.png", "Correlação de Spearman vs. n_perturbations"),
        "",
        _fig_ref("q11_spearman_violin.png", "Violino: Spearman por n_perturbations"),
        "",
        _fig_ref("q11_top5_overlap_heatmap.png", "Top-5 Overlap: instância × n_perturbations"),
        "",
        _fig_ref("q11_variance_vs_overlap_scatter.png", "Variância vs. Top-5 Overlap"),
        "",
    ]

    if has_q11:
        lines += [
            "**Evolução do Spearman médio por n:**",
            "",
            _table_md(n_level[["n_perturbations", "spearman_mean", "spearman_std",
                                 "overlap_mean", "elapsed_mean"]].round(4)),
            "",
        ]

    lines += [
        "### 4.2 Custo Computacional",
        "",
        _fig_ref("q11_cost_vs_n.png", "Custo Computacional (segundos) × n_perturbations"),
        "",
    ]

    lines += [
        "### 4.3 n Adaptativo",
        "",
        _fig_ref("q11_adaptive_n_histogram.png", "Histograma do n Adaptativo por Instância"),
        "",
        _fig_ref("q11_adaptive_n_boxplot.png", "Boxplot do n Adaptativo"),
        "",
    ]

    if has_q11:
        lines += [
            f"**Mediana do n adaptativo encontrado:** `{median_n}`",
            f"(critério: Spearman ≥ {SPEARMAN_THRESHOLD} entre n e 2n)",
            "",
        ]

    lines += [
        "### 4.4 Features Estáveis",
        "",
        _fig_ref("q11_stable_features_barplot.png", "Features Mais Frequentes no Top-5 (Referência)"),
        "",
        _fig_ref("q11_metric_correlation_heatmap.png", "Correlação entre Métricas do Experimento"),
        "",
    ]

    if slm_available:
        lines += [
            "### 4.5 Coerência Textual (LLM-as-judge)",
            "",
            _fig_ref("q11_llm_judge_vs_n.png", "Coerência Textual (LLM-as-judge) × n_perturbations"),
            "",
            _fig_ref("q11_coherence_vs_spearman_scatter.png", "Coerência vs. Spearman"),
            "",
        ]

    # ── 5. Exemplos de Explicação ──────────────────────────────────────────────
    if base_texts is not None and slm_available:
        lines += [
            "## 5. Exemplos de Explicações Textuais (LIME Base)",
            "",
        ]
        for _, row in base_texts.iterrows():
            if row.get("llm_status") == "ok":
                lines += [
                    f"**Instância — {row['tipo']}:**",
                    f"> {row['explanation_text'][:600]}",
                    "",
                ]

    # ── 6. Discussão ──────────────────────────────────────────────────────────
    lines += [
        "## 6. Discussão",
        "",
        "### Limitações",
        "",
        "- A avaliação textual depende da disponibilidade do SLM local; sessões offline produzem apenas métricas numéricas.",
        "- O experimento usa 30 instâncias de teste; maior cobertura pode alterar o `n` mediano.",
        "- A correlação de Spearman não captura diferenças nos valores absolutos, apenas no ranking.",
        "- O limiar de 0.85 foi definido heuristicamente; análise de sensibilidade seria recomendável.",
        "",
        "### Comparação com a Literatura",
        "",
        "Zeng & Zhu (2024) demonstraram que a verbalização de atribuições numéricas via LLMs "
        "melhora a interpretabilidade em cenários de domínio específico. Os resultados deste "
        "experimento complementam essa abordagem ao investigar o custo-benefício do n de perturbações.",
        "",
        "### Trabalhos Futuros",
        "",
        "- Estender para outros datasets de crédito com mais features.",
        "- Avaliar com múltiplos modelos (XGBoost, Logistic Regression).",
        "- Aplicar `n` adaptativo na Questão 1.2 (ensemble de explicações pobres).",
        "- Explorar SLMs maiores para verificar se a coerência textual melhora.",
        "",
    ]

    # ── 7. Conclusão ──────────────────────────────────────────────────────────
    lines += [
        "## 7. Conclusão",
        "",
        "**Resposta à Questão 1.1:**",
        "",
        f"Para o dataset de risco de crédito e o modelo RandomForestClassifier, "
        f"o menor número de perturbações do LIME que produz explicações estáveis "
        f"(Spearman ≥ {SPEARMAN_THRESHOLD} entre n e 2n) é, em mediana, "
        f"**n = {median_n}** sobre as {30} instâncias de teste avaliadas.",
        "",
        "A estratégia adaptativa de busca sequencial com early stopping demonstra ser eficaz "
        "para evitar cálculos desnecessários com valores altos de n quando o limiar de "
        "estabilidade é atingido cedo.",
        "",
        "---",
        "",
        "*Relatório gerado automaticamente por `scripts/04_relatorio_final.py`*",
    ]

    # ── Escreve arquivo ───────────────────────────────────────────────────────
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Relatório salvo: %s", REPORT_MD)
    logger.info("=== Relatório Final | Concluído ===")


if __name__ == "__main__":
    run_report()
