"""
02_lime_base.py — Valida o pipeline LIME → tuplas → SLM com 3 instâncias.
"""

import logging
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BASELINE_PREDICTIONS_CSV, PREPROCESSOR_PATH, RF_MODEL_PATH,
    LIME_BASE_N_SAMPLES, RANDOM_STATE, TOP_K,
    LIME_BASE_INSTANCES_CSV, LIME_BASE_CONTRIBUTIONS_CSV, LIME_BASE_TEXTS_CSV,
    FIG_DIR,
)
from src.io_utils import load_csv, save_csv, save_fig
from src.explainer import compute_lime_explanation, contributions_to_label
from src.llm_client import explain_with_llm, is_slm_available
import src.plotting as P
from scripts.data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def select_representative_instances(
    X_test: np.ndarray,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> list[dict]:
    """Seleciona 1 VP, 1 VN e 1 FP (índices no array de teste)."""
    y_test_arr = np.array(y_test)
    vp_idx = np.where((y_test_arr == 1) & (y_pred == 1))[0]
    vn_idx = np.where((y_test_arr == 0) & (y_pred == 0))[0]
    fp_idx = np.where((y_test_arr == 0) & (y_pred == 1))[0]

    def best(idxs, maximize: bool = True) -> int:
        if len(idxs) == 0:
            raise ValueError("Sem exemplos para o critério solicitado.")
        probs = y_proba[idxs]
        return idxs[np.argmax(probs) if maximize else np.argmin(probs)]

    return [
        {"tipo": "verdadeiro_positivo", "test_idx": int(best(vp_idx, True))},
        {"tipo": "verdadeiro_negativo", "test_idx": int(best(vn_idx, False))},
        {"tipo": "falso_positivo", "test_idx": int(best(fp_idx, True))},
    ]


def run_lime_base() -> list[str | None]:
    logger.info("=== LIME Base | Início ===")

    model = joblib.load(RF_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test_t, y_test, feature_names, X_train_t = load_processed_data(preprocessor)

    preds_df = load_csv(BASELINE_PREDICTIONS_CSV)
    y_pred = preds_df["y_pred"].values
    y_proba = preds_df["y_proba"].values

    instances_meta = select_representative_instances(X_test_t, y_test.reset_index(drop=True), y_pred, y_proba)
    instances_df = pd.DataFrame(instances_meta)
    save_csv(instances_df, LIME_BASE_INSTANCES_CSV)

    contributions_rows = []
    text_rows = []
    baseline_explanations: list[str | None] = []
    feature_counter: Counter = Counter()

    for meta in instances_meta:
        idx = meta["test_idx"]
        instance = X_test_t[idx]
        tipo = meta["tipo"]

        logger.info("Gerando LIME para instância %s (idx=%d)...", tipo, idx)
        contribs, elapsed = compute_lime_explanation(
            model, instance, X_train_t, feature_names,
            n_samples=LIME_BASE_N_SAMPLES, random_state=RANDOM_STATE,
        )

        for rank, (feat, val) in enumerate(contribs):
            contributions_rows.append({
                "tipo": tipo, "test_idx": idx, "rank": rank,
                "feature": feat, "lime_value": val,
            })
            if rank < TOP_K:
                feature_counter[feat] += 1

        prob = float(y_proba[idx])
        label = contributions_to_label(contribs, prob)
        text = explain_with_llm(contribs[:TOP_K], label)

        text_rows.append({
            "tipo": tipo, "test_idx": idx,
            "prediction_label": label,
            "explanation_text": text or "[SLM indisponível]",
            "llm_status": "ok" if text else "offline_skip",
        })
        baseline_explanations.append(text)

        # Gráfico individual
        fig = P.lime_contributions_bar(contribs[:10], f"LIME — {tipo} (idx={idx})")
        save_fig(fig, FIG_DIR / f"lime_base_{tipo}.png")
        # Compatível com nome sem sufixo _1, _2, _3
        save_fig(
            P.lime_contributions_bar(contribs[:10], f"LIME — {tipo} (idx={idx})"),
            FIG_DIR / f"lime_base_instance_{['verdadeiro_positivo','verdadeiro_negativo','falso_positivo'].index(tipo)+1}.png"
        )

    save_csv(pd.DataFrame(contributions_rows), LIME_BASE_CONTRIBUTIONS_CSV)
    save_csv(pd.DataFrame(text_rows), LIME_BASE_TEXTS_CSV)

    # Gráfico de frequência de features no top-5
    freq_df = pd.DataFrame(feature_counter.most_common(), columns=["feature", "count"])
    fig = P.feature_frequency_bar(freq_df)
    save_fig(fig, FIG_DIR / "lime_base_feature_frequency.png")

    logger.info("SLM disponível: %s", is_slm_available())
    logger.info("=== LIME Base | Concluído ===")
    return baseline_explanations


if __name__ == "__main__":
    run_lime_base()
