"""
evaluation.py — Métricas numéricas de estabilidade e coerência das explicações LIME.
"""

import logging

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _rank_vector(contributions: list[tuple[str, float]]) -> tuple[list[str], list[float]]:
    """Extrai features e escores de uma lista de contribuições."""
    features = [f for f, _ in contributions]
    scores = [v for _, v in contributions]
    return features, scores


def spearman_stability(
    explanations_a: list[tuple[str, float]],
    explanations_b: list[tuple[str, float]],
) -> float:
    """
    Correlação de Spearman entre os escores de duas explicações LIME.
    Alinha pelo conjunto de features em comum.
    Retorna valor entre -1 e 1.
    """
    dict_a = dict(explanations_a)
    dict_b = dict(explanations_b)
    common = sorted(set(dict_a) & set(dict_b))
    if len(common) < 2:
        return 0.0
    vec_a = [dict_a[f] for f in common]
    vec_b = [dict_b[f] for f in common]
    corr, _ = spearmanr(vec_a, vec_b)
    return float(corr) if not np.isnan(corr) else 0.0


def top_k_overlap(
    explanation_ref: list[tuple[str, float]],
    explanation_test: list[tuple[str, float]],
    k: int = 5,
) -> float:
    """
    Fração de features em comum no top-k entre referência e teste.
    Retorna valor em [0, 1].
    """
    top_ref = {f for f, _ in sorted(explanation_ref, key=lambda x: abs(x[1]), reverse=True)[:k]}
    top_test = {f for f, _ in sorted(explanation_test, key=lambda x: abs(x[1]), reverse=True)[:k]}
    if not top_ref:
        return 0.0
    return len(top_ref & top_test) / k


def lime_score_variance(
    explanations: list[list[tuple[str, float]]],
    feature_name: str,
) -> float:
    """
    Variância dos scores LIME de uma feature específica entre N execuções.
    Features ausentes em alguma execução são ignoradas.
    """
    scores = []
    for exp in explanations:
        d = dict(exp)
        if feature_name in d:
            scores.append(d[feature_name])
    if len(scores) < 2:
        return 0.0
    return float(np.var(scores))


def mean_spearman_across_repetitions(
    repetitions: list[list[tuple[str, float]]],
) -> float:
    """
    Calcula a Spearman média entre todos os pares únicos de repetições.
    """
    n = len(repetitions)
    if n < 2:
        return 1.0
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(spearman_stability(repetitions[i], repetitions[j]))
    return float(np.mean(scores))


def adaptive_n_perturbations(
    model,
    instance: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    spearman_threshold: float = 0.85,
    n_candidates: list[int] | None = None,
    categorical_features: list[int] | None = None,
    base_seed: int = 42,
) -> int:
    """
    Retorna o menor n tal que Spearman(n, 2n) >= spearman_threshold.
    Busca sequencial com early stopping.

    Se nenhum n satisfaz o critério, retorna o maior candidato.
    """
    from src.explainer import compute_lime_explanation

    if n_candidates is None:
        n_candidates = [50, 100, 200, 500, 1000]

    for idx, n in enumerate(n_candidates[:-1]):
        n_next = n_candidates[idx + 1]
        exp_n, _ = compute_lime_explanation(model, instance, X_train, feature_names, n, base_seed)
        exp_next, _ = compute_lime_explanation(model, instance, X_train, feature_names, n_next, base_seed + 1)
        corr = spearman_stability(exp_n, exp_next)
        logger.debug("n=%d → n_next=%d → Spearman=%.4f", n, n_next, corr)
        if corr >= spearman_threshold:
            return n

    return n_candidates[-1]
