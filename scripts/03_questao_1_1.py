"""
03_questao_1_1.py — Experimento principal Q1.1.
Varia n_perturbations do LIME e determina o n adaptativo mínimo para
explicações estáveis, com métricas numéricas e coerência via SLM.
"""

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    RF_MODEL_PATH, PREPROCESSOR_PATH,
    LIME_N_PERTURBATIONS_GRID, LIME_REFERENCE_N, LIME_N_INSTANCES_Q11,
    LIME_REPETITIONS, RANDOM_STATE, SPEARMAN_THRESHOLD, TOP_K,
    Q11_RAW_CSV, Q11_INSTANCE_CSV, Q11_N_LEVEL_CSV,
    Q11_ADAPTIVE_CSV, Q11_TEXTS_CSV, Q11_CONCLUSIONS_CSV,
    BASELINE_PREDICTIONS_CSV,
    FIG_DIR,
)
from src.io_utils import load_csv, save_csv, save_fig, append_row_csv
from src.explainer import compute_lime_explanation, contributions_to_label
from src.evaluation import (
    spearman_stability, top_k_overlap, lime_score_variance,
    mean_spearman_across_repetitions,
)
from src.llm_client import explain_with_llm, llm_coherence_score, is_slm_available
import src.plotting as P
from scripts.data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def select_instances_by_decile(
    X_test: np.ndarray,
    y_proba: np.ndarray,
    n_total: int = 30,
) -> np.ndarray:
    """Seleciona n_total instâncias de forma uniforme por decis de probabilidade."""
    n_deciles = 10
    per_decile = max(1, n_total // n_deciles)
    bins = np.percentile(y_proba, np.linspace(0, 100, n_deciles + 1))
    selected_idxs = []
    rng = np.random.default_rng(RANDOM_STATE)
    for i in range(n_deciles):
        mask = (y_proba >= bins[i]) & (y_proba <= bins[i + 1])
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            chosen = rng.choice(idxs, size=min(per_decile, len(idxs)), replace=False)
            selected_idxs.extend(chosen.tolist())
    # remove duplicatas e limita
    seen = set()
    unique = []
    for idx in selected_idxs:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return np.array(unique[:n_total])


def run_q11() -> None:
    logger.info("=== Q1.1 | Início ===")
    logger.info("SLM disponível: %s", is_slm_available())

    model = joblib.load(RF_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test_t, y_test, feature_names, X_train_t = load_processed_data(preprocessor)

    preds_df = load_csv(BASELINE_PREDICTIONS_CSV)
    y_proba_all = preds_df["y_proba"].values
    y_pred_all = preds_df["y_pred"].values

    # ── Seleciona instâncias ──────────────────────────────────────────────────
    instance_idxs = select_instances_by_decile(X_test_t, y_proba_all, LIME_N_INSTANCES_Q11)
    logger.info("%d instâncias selecionadas", len(instance_idxs))

    # ── Referências com n=1000 ────────────────────────────────────────────────
    logger.info("Calculando explicações de referência (n=%d)...", LIME_REFERENCE_N)
    reference_explanations: dict[int, list[tuple[str, float]]] = {}
    for idx in tqdm(instance_idxs, desc="Referências"):
        exp, _ = compute_lime_explanation(
            model, X_test_t[idx], X_train_t, feature_names,
            n_samples=LIME_REFERENCE_N, random_state=RANDOM_STATE,
        )
        reference_explanations[int(idx)] = exp

    # ── Loop principal ────────────────────────────────────────────────────────
    logger.info("Iniciando grid de experimentos...")
    total_runs = len(instance_idxs) * len(LIME_N_PERTURBATIONS_GRID) * LIME_REPETITIONS
    pbar = tqdm(total=total_runs, desc="Runs Q1.1")

    for inst_num, test_idx in enumerate(instance_idxs):
        test_idx = int(test_idx)
        instance = X_test_t[test_idx]
        prob = float(y_proba_all[test_idx])
        ref_exp = reference_explanations[test_idx]

        for n in LIME_N_PERTURBATIONS_GRID:
            reps: list[list[tuple[str, float]]] = []
            elapsed_list: list[float] = []

            for rep in range(LIME_REPETITIONS):
                seed = RANDOM_STATE + rep
                exp, elapsed = compute_lime_explanation(
                    model, instance, X_train_t, feature_names,
                    n_samples=n, random_state=seed,
                )
                reps.append(exp)
                elapsed_list.append(elapsed)

                # Métricas por run
                spearman_vs_ref = spearman_stability(exp, ref_exp)
                overlap_vs_ref = top_k_overlap(ref_exp, exp, TOP_K)

                # Texto e coerência via SLM
                label = contributions_to_label(exp, prob)
                text = explain_with_llm(exp[:TOP_K], label)
                coherence = llm_coherence_score(text) if text else None

                row = {
                    "inst_num": inst_num,
                    "test_idx": test_idx,
                    "n_perturbations": n,
                    "repetition": rep,
                    "spearman_vs_ref": spearman_vs_ref,
                    "overlap_vs_ref": overlap_vs_ref,
                    "elapsed_sec": elapsed,
                    "llm_coherence": coherence,
                    "llm_status": "ok" if text else "offline_skip",
                }
                append_row_csv(row, Q11_RAW_CSV)

                if text:
                    append_row_csv(
                        {"inst_num": inst_num, "test_idx": test_idx,
                         "n_perturbations": n, "repetition": rep,
                         "prediction_label": label, "text": text},
                        Q11_TEXTS_CSV,
                    )

                pbar.update(1)

            # Métricas agregadas por (instância, n)
            mean_spearman_reps = mean_spearman_across_repetitions(reps)
            variances = [lime_score_variance(reps, feat) for feat, _ in reps[0][:TOP_K]]
            mean_variance = float(np.mean(variances)) if variances else 0.0

    pbar.close()

    # ── Agregações ────────────────────────────────────────────────────────────
    logger.info("Calculando agregações...")
    raw = pd.read_csv(Q11_RAW_CSV)

    # Por instância × n
    inst_n = (
        raw.groupby(["test_idx", "n_perturbations"])
        .agg(
            spearman_mean=("spearman_vs_ref", "mean"),
            spearman_std=("spearman_vs_ref", "std"),
            overlap_mean=("overlap_vs_ref", "mean"),
            overlap_std=("overlap_vs_ref", "std"),
            elapsed_mean=("elapsed_sec", "mean"),
            elapsed_std=("elapsed_sec", "std"),
            coherence_mean=("llm_coherence", "mean"),
        )
        .reset_index()
    )
    save_csv(inst_n, Q11_INSTANCE_CSV)

    # Por nível de n
    n_level = (
        raw.groupby("n_perturbations")
        .agg(
            spearman_mean=("spearman_vs_ref", "mean"),
            spearman_std=("spearman_vs_ref", "std"),
            overlap_mean=("overlap_vs_ref", "mean"),
            overlap_std=("overlap_vs_ref", "std"),
            elapsed_mean=("elapsed_sec", "mean"),
            elapsed_std=("elapsed_sec", "std"),
            coherence_mean=("llm_coherence", "mean"),
        )
        .reset_index()
        .sort_values("n_perturbations")
    )
    save_csv(n_level, Q11_N_LEVEL_CSV)

    # ── n adaptativo por instância ────────────────────────────────────────────
    logger.info("Determinando n adaptativo por instância...")
    adaptive_rows = []
    for test_idx in instance_idxs:
        test_idx = int(test_idx)
        instance = X_test_t[test_idx]
        adaptive_n = None
        n_grid = LIME_N_PERTURBATIONS_GRID
        for i, n in enumerate(n_grid[:-1]):
            row_n = inst_n[(inst_n["test_idx"] == test_idx) & (inst_n["n_perturbations"] == n)]
            row_2n = inst_n[(inst_n["test_idx"] == test_idx) & (inst_n["n_perturbations"] == n_grid[i + 1])]
            if row_n.empty or row_2n.empty:
                continue
            s_n = float(row_n["spearman_mean"].iloc[0])
            s_2n = float(row_2n["spearman_mean"].iloc[0])
            if min(s_n, s_2n) >= SPEARMAN_THRESHOLD:
                adaptive_n = n
                break
        if adaptive_n is None:
            adaptive_n = n_grid[-1]
        adaptive_rows.append({"test_idx": test_idx, "adaptive_n": adaptive_n})

    adaptive_df = pd.DataFrame(adaptive_rows)
    save_csv(adaptive_df, Q11_ADAPTIVE_CSV)
    median_n = int(adaptive_df["adaptive_n"].median())
    print(f"n adaptativo recomendado: {median_n} (mediana sobre as {len(instance_idxs)} instâncias)")

    # ── Conclusões ────────────────────────────────────────────────────────────
    best_n_by_spearman = n_level.loc[n_level["spearman_mean"].idxmax(), "n_perturbations"]
    conclusions = pd.DataFrame([
        {"metrica": "n_adaptativo_mediana", "valor": median_n},
        {"metrica": "n_melhor_spearman", "valor": int(best_n_by_spearman)},
        {"metrica": "spearman_n1000", "valor": float(n_level[n_level["n_perturbations"] == 1000]["spearman_mean"].iloc[0])},
        {"metrica": "spearman_n25", "valor": float(n_level[n_level["n_perturbations"] == 25]["spearman_mean"].iloc[0])},
        {"metrica": "threshold_usado", "valor": SPEARMAN_THRESHOLD},
    ])
    save_csv(conclusions, Q11_CONCLUSIONS_CSV)

    # ── Gráficos ──────────────────────────────────────────────────────────────
    logger.info("Gerando gráficos...")

    fig = P.line_mean_std(
        n_level["n_perturbations"].tolist(),
        n_level["spearman_mean"].tolist(),
        n_level["spearman_std"].fillna(0).tolist(),
        xlabel="n_perturbations",
        ylabel="Spearman vs. Referência (n=1000)",
        title="Estabilidade LIME: Spearman × n_perturbations",
    )
    save_fig(fig, FIG_DIR / "q11_spearman_vs_n.png")

    if raw["llm_coherence"].notna().any():
        fig = P.line_mean_std(
            n_level["n_perturbations"].tolist(),
            n_level["coherence_mean"].fillna(0).tolist(),
            [0.0] * len(n_level),
            xlabel="n_perturbations",
            ylabel="Coerência LLM-as-judge (1–5)",
            title="Coerência Textual × n_perturbations",
            color="darkorange",
        )
        save_fig(fig, FIG_DIR / "q11_llm_judge_vs_n.png")

    # Heatmap top-5 overlap: instância × n
    pivot = inst_n.pivot(index="test_idx", columns="n_perturbations", values="overlap_mean").fillna(0)
    fig = P.overlap_heatmap(pivot, "Top-5 Overlap (instância × n_perturbations)")
    save_fig(fig, FIG_DIR / "q11_top5_overlap_heatmap.png")

    # Boxplot do n adaptativo
    fig = P.boxplot_series(adaptive_df["adaptive_n"], "Distribuição do n Adaptativo",
                            "Instâncias", "n_perturbations")
    save_fig(fig, FIG_DIR / "q11_adaptive_n_boxplot.png")

    # Violino Spearman por n
    fig = P.violin_by_n(raw, "n_perturbations", "spearman_vs_ref",
                         "Violino: Spearman por n_perturbations")
    save_fig(fig, FIG_DIR / "q11_spearman_violin.png")

    # Histograma n adaptativo
    fig = P.histogram_adaptive_n(adaptive_df["adaptive_n"])
    save_fig(fig, FIG_DIR / "q11_adaptive_n_histogram.png")

    # Features mais estáveis (top-k frequentes na referência)
    all_top_feats = []
    for exp in reference_explanations.values():
        top_feats = [f for f, _ in sorted(exp, key=lambda x: abs(x[1]), reverse=True)[:TOP_K]]
        all_top_feats.extend(top_feats)
    from collections import Counter
    freq_counter = Counter(all_top_feats)
    freq_df = pd.DataFrame(freq_counter.most_common(15), columns=["feature", "count"])
    fig = P.feature_frequency_bar(freq_df, "Features Mais Frequentes no Top-5 (Referência n=1000)")
    save_fig(fig, FIG_DIR / "q11_stable_features_barplot.png")

    # Scatter: coerência vs. Spearman
    if raw["llm_coherence"].notna().any():
        fig = P.scatter_two_metrics(
            raw["spearman_vs_ref"], raw["llm_coherence"].fillna(0),
            "Spearman vs. Referência", "Coerência LLM (1–5)",
            "Coerência Textual vs. Estabilidade Numérica",
        )
        save_fig(fig, FIG_DIR / "q11_coherence_vs_spearman_scatter.png")

    # Scatter: variância vs. overlap
    raw_agg = raw.groupby(["test_idx", "n_perturbations"])["spearman_vs_ref"].var().reset_index()
    raw_agg.columns = ["test_idx", "n_perturbations", "spearman_var"]
    merged = inst_n.merge(raw_agg, on=["test_idx", "n_perturbations"], how="left")
    fig = P.scatter_two_metrics(
        merged["spearman_var"].fillna(0), merged["overlap_mean"],
        "Variância do Spearman", "Top-5 Overlap",
        "Variância vs. Overlap por (instância, n)",
    )
    save_fig(fig, FIG_DIR / "q11_variance_vs_overlap_scatter.png")

    # Custo computacional
    fig = P.cost_vs_n(
        n_level["n_perturbations"].tolist(),
        n_level["elapsed_mean"].tolist(),
        n_level["elapsed_std"].fillna(0).tolist(),
    )
    save_fig(fig, FIG_DIR / "q11_cost_vs_n.png")

    # Correlação entre métricas numéricas
    metric_cols = ["n_perturbations", "spearman_mean", "overlap_mean", "elapsed_mean"]
    available = [c for c in metric_cols if c in inst_n.columns]
    fig = P.metric_correlation_heatmap(inst_n[available], "Correlação entre Métricas (Q1.1)")
    save_fig(fig, FIG_DIR / "q11_metric_correlation_heatmap.png")

    logger.info("=== Q1.1 | Concluído ===")


if __name__ == "__main__":
    run_q11()
