"""
tests.py — Verificações de integridade do ambiente e dos artefatos gerados.
Uso: python tests.py
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PASS = "✓"
FAIL = "✗"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Registra no log se uma verificação passou ou falhou."""
    if condition:
        logger.info("%s %s", PASS, name)
    else:
        msg = f"{name}" + (f" — {detail}" if detail else "")
        logger.error("%s %s", FAIL, msg)
        _failures.append(msg)


def check_csv_columns(path: Path, required_cols: list[str]) -> None:
    """Confere se um CSV existe e contém as colunas mínimas esperadas."""
    import pandas as pd
    name = f"Colunas em {path.name}"
    if not path.exists():
        check(name, False, "arquivo não encontrado")
        return
    df = pd.read_csv(path, nrows=1)
    missing = [c for c in required_cols if c not in df.columns]
    check(name, len(missing) == 0, f"colunas ausentes: {missing}")


def run_tests() -> None:
    """Executa as validações do ambiente e dos artefatos do projeto.

    Observação: antes de rodar `main.py`, é esperado que falhem os testes que
    dependem de modelos, CSVs, figuras e relatório ainda não gerados.
    """
    from src.config import (
        DATA_PATH, RF_MODEL_PATH, PREPROCESSOR_PATH,
        BASELINE_METRICS_CSV, BASELINE_PREDICTIONS_CSV, BASELINE_IMPORTANCE_CSV,
        Q11_RAW_CSV, Q11_ADAPTIVE_CSV, Q11_CONCLUSIONS_CSV,
        EDA_MISSINGNESS_CSV, EDA_NUMERIC_CSV,
        REPORT_MD, FIG_DIR, MODELS_DIR, CSV_DIR,
        MIN_AUC_ROC,
    )

    # Limpa o histórico para permitir múltiplas execuções no mesmo processo.
    _failures.clear()

    logger.info("=" * 55)
    logger.info("TESTS — Verificando ambiente e artefatos")
    logger.info("=" * 55)

    # ── Ambiente Python ───────────────────────────────────────────────────────
    check("Python >= 3.10", sys.version_info >= (3, 10),
          f"versão atual: {sys.version_info.major}.{sys.version_info.minor}")

    # Estes pacotes são o núcleo do pipeline: dados, modelo, métricas,
    # visualização e serialização.
    for pkg in ["pandas", "numpy", "sklearn", "lime", "scipy", "matplotlib", "seaborn", "joblib"]:
        try:
            __import__(pkg)
            check(f"Pacote '{pkg}' instalado", True)
        except ImportError:
            check(f"Pacote '{pkg}' instalado", False, "não encontrado")

    # ── Dataset ───────────────────────────────────────────────────────────────
    check("Dataset existe", DATA_PATH.exists(), str(DATA_PATH))
    if DATA_PATH.exists():
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        check("Dataset tem >= 1000 linhas", len(df) >= 1000, f"{len(df)} linhas")
        check("Coluna target 'loan_status' presente", "loan_status" in df.columns)

    # ── Diretórios ────────────────────────────────────────────────────────────
    for d in [MODELS_DIR, CSV_DIR, FIG_DIR]:
        check(f"Diretório {d.name}/ existe", d.exists())

    # ── Modelos serializados ──────────────────────────────────────────────────
    check("rf_model.joblib existe", RF_MODEL_PATH.exists())
    check("preprocessor.joblib existe", PREPROCESSOR_PATH.exists())

    # ── CSVs da EDA ───────────────────────────────────────────────────────────
    check("EDA: missingness CSV existe", EDA_MISSINGNESS_CSV.exists())
    check("EDA: numeric summary CSV existe", EDA_NUMERIC_CSV.exists())
    check_csv_columns(EDA_MISSINGNESS_CSV, ["feature", "missing_count", "missing_pct"])

    # ── CSVs do Baseline ──────────────────────────────────────────────────────
    check("Baseline: metrics CSV existe", BASELINE_METRICS_CSV.exists())
    check("Baseline: predictions CSV existe", BASELINE_PREDICTIONS_CSV.exists())
    check("Baseline: feature importance CSV existe", BASELINE_IMPORTANCE_CSV.exists())
    check_csv_columns(BASELINE_METRICS_CSV, ["auc_roc", "f1_macro", "accuracy"])
    check_csv_columns(BASELINE_PREDICTIONS_CSV, ["y_true", "y_pred", "y_proba"])

    if BASELINE_METRICS_CSV.exists():
        import pandas as pd
        m = pd.read_csv(BASELINE_METRICS_CSV)
        auc = float(m["auc_roc"].iloc[0])
        # O baseline precisa atingir um patamar mínimo para que a análise de
        # explicabilidade faça sentido sobre um modelo razoavelmente útil.
        check(f"AUC-ROC >= {MIN_AUC_ROC}", auc >= MIN_AUC_ROC, f"valor atual: {auc:.4f}")

    # ── CSVs do Q1.1 ─────────────────────────────────────────────────────────
    check("Q1.1: raw runs CSV existe", Q11_RAW_CSV.exists())
    check("Q1.1: adaptive n CSV existe", Q11_ADAPTIVE_CSV.exists())
    check("Q1.1: conclusions CSV existe", Q11_CONCLUSIONS_CSV.exists())
    check_csv_columns(Q11_RAW_CSV, [
        "test_idx", "n_perturbations", "repetition",
        "spearman_vs_ref", "overlap_vs_ref", "elapsed_sec",
    ])
    check_csv_columns(Q11_ADAPTIVE_CSV, ["test_idx", "adaptive_n"])

    if Q11_ADAPTIVE_CSV.exists():
        import pandas as pd
        adp = pd.read_csv(Q11_ADAPTIVE_CSV)
        check("Q1.1: >= 10 instâncias no adaptive_n", len(adp) >= 10, f"{len(adp)} linhas")
        check("Q1.1: adaptive_n não-nulo", adp["adaptive_n"].notna().all())

    # ── Figuras ───────────────────────────────────────────────────────────────
    expected_figs = [
        "eda_target_distribution.png",
        "eda_correlation_heatmap.png",
        "baseline_roc_curve.png",
        "baseline_confusion_matrix.png",
        "baseline_feature_importance_top15.png",
        "q11_spearman_vs_n.png",
        "q11_top5_overlap_heatmap.png",
        "q11_adaptive_n_histogram.png",
        "q11_spearman_violin.png",
        "q11_cost_vs_n.png",
    ]
    for fig in expected_figs:
        check(f"Figura {fig} existe", (FIG_DIR / fig).exists())

    total_figs = len(list(FIG_DIR.glob("*.png"))) if FIG_DIR.exists() else 0
    check("Pelo menos 10 figuras geradas", total_figs >= 10, f"{total_figs} encontradas")

    # ── Relatório ─────────────────────────────────────────────────────────────
    check("relatorio_final.md existe", REPORT_MD.exists())
    if REPORT_MD.exists():
        content = REPORT_MD.read_text(encoding="utf-8")
        check("Relatório contém 'n adaptativo'", "n adaptativo" in content.lower())
        check("Relatório referência figuras (.png)", ".png)" in content)

    # ── Resultado final ───────────────────────────────────────────────────────
    logger.info("=" * 55)
    if _failures:
        # Mantemos saída com código 1 para facilitar uso em automação e CI.
        logger.error("%d falha(s) encontrada(s):", len(_failures))
        for f in _failures:
            logger.error("  • %s", f)
        sys.exit(1)
    else:
        logger.info("Todos os testes passaram ✓")


if __name__ == "__main__":
    run_tests()
