"""
explainer.py — Orquestra LIME → tuplas → LLM → texto.
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_lime_explanation(
    model: Any,
    instance: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    n_samples: int,
    random_state: int = 42,
    categorical_features: list[int] | None = None,
) -> tuple[list[tuple[str, float]], float]:
    """
    Gera explicação LIME para uma instância.

    Retorna:
        contributions : lista de (feature_name, lime_value) ordenada por |valor| desc
        elapsed_sec   : tempo de geração em segundos
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="classification",
        categorical_features=categorical_features or [],
        random_state=random_state,
        verbose=False,
    )

    t0 = time.perf_counter()
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=len(feature_names),
        num_samples=n_samples,
    )
    elapsed = time.perf_counter() - t0

    raw = exp.as_list()
    # Mapear para nomes originais: LIME pode truncar nomes com condições (">")
    contributions = sorted(
        [(feat, val) for feat, val in raw],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return contributions, elapsed


def contributions_to_label(contributions: list[tuple[str, float]], prediction_proba: float) -> str:
    """Converte probabilidade em rótulo textual legível."""
    if prediction_proba >= 0.5:
        return f"inadimplente (probabilidade {prediction_proba:.2%})"
    return f"não inadimplente (probabilidade {1 - prediction_proba:.2%})"
