"""
llm_client.py — Interface com o Docker Model Runner (ai/qwen2.5).
Suporta cache local e modo offline gracioso.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.config import (
    LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P,
    LLM_PROMPTS_CSV, LLM_RESPONSES_JSON,
)
from src.io_utils import append_row_csv, load_json, save_json

logger = logging.getLogger(__name__)

_SLM_AVAILABLE: bool | None = None  # None = ainda não verificado


def _hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    return load_json(LLM_RESPONSES_JSON)


def _save_to_cache(prompt_hash: str, call_type: str, response: str | None, status: str) -> None:
    cache = _load_cache()
    cache[prompt_hash] = {
        "call_type": call_type,
        "model": LLM_MODEL,
        "timestamp": datetime.utcnow().isoformat(),
        "response": response,
        "status": status,
    }
    save_json(cache, LLM_RESPONSES_JSON)
    append_row_csv(
        {
            "prompt_hash": prompt_hash,
            "call_type": call_type,
            "model": LLM_MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
        },
        LLM_PROMPTS_CSV,
    )


def is_slm_available() -> bool:
    """Verifica disponibilidade do Docker Model Runner (uma vez por processo)."""
    global _SLM_AVAILABLE
    if _SLM_AVAILABLE is not None:
        return _SLM_AVAILABLE
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key="na")
        client.models.list()
        _SLM_AVAILABLE = True
        logger.info("Docker Model Runner disponível em %s", LLM_BASE_URL)
    except Exception as exc:
        _SLM_AVAILABLE = False
        logger.warning("SLM indisponível (%s). Etapas textuais serão puladas.", exc)
    return _SLM_AVAILABLE


def _call_slm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key="na")
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        top_p=LLM_TOP_P,
    )
    return response.choices[0].message.content.strip()


def explain_with_llm(
    feature_contributions: list[tuple[str, float]],
    prediction: str,
    domain_context: str = "risco de crédito",
    model: str = LLM_MODEL,
) -> str | None:
    """
    Gera explicação textual a partir das contribuições LIME.
    Retorna None se o SLM estiver offline.
    Usa cache local para evitar chamadas repetidas.
    """
    tuples_str = "\n".join(f"  - {feat}: {val:+.4f}" for feat, val in feature_contributions)
    prompt = (
        f"Você é um assistente especialista em {domain_context}. "
        f"O modelo de machine learning previu '{prediction}' para a seguinte "
        f"solicitação de empréstimo. As contribuições de cada atributo para essa previsão são:\n"
        f"{tuples_str}\n"
        f"Explique em linguagem simples, sem jargão técnico, por que o modelo chegou "
        f"a essa conclusão. Seja objetivo e use no máximo 3 parágrafos."
    )
    prompt_hash = _hash_prompt(prompt)
    cache = _load_cache()

    if prompt_hash in cache:
        logger.debug("Cache hit: %s", prompt_hash)
        _save_to_cache(prompt_hash, "explicacao", cache[prompt_hash]["response"], "cache_hit")
        return cache[prompt_hash]["response"]

    if not is_slm_available():
        _save_to_cache(prompt_hash, "explicacao", None, "offline_skip")
        return None

    try:
        response = _call_slm(prompt)
        _save_to_cache(prompt_hash, "explicacao", response, "cache_miss")
        return response
    except Exception as exc:
        logger.warning("Erro ao chamar SLM: %s", exc)
        _save_to_cache(prompt_hash, "explicacao", None, "offline_skip")
        return None


def llm_coherence_score(
    explanation_text: str,
    evaluator_model: str = LLM_MODEL,
) -> int | None:
    """
    Avalia coerência de uma explicação em escala 1–5 usando o próprio SLM.
    Retorna None se o SLM estiver offline.
    """
    prompt = (
        "Avalie a coerência e clareza da seguinte explicação de um modelo de machine learning "
        "para um leigo. Responda APENAS com um JSON no formato {\"score\": N} onde N é um "
        "inteiro de 1 (péssimo) a 5 (excelente). Não acrescente nenhum texto extra.\n\n"
        f"Explicação:\n{explanation_text}"
    )
    prompt_hash = _hash_prompt(prompt + "_judge")
    cache = _load_cache()

    if prompt_hash in cache:
        raw = cache[prompt_hash]["response"]
        if raw:
            try:
                return int(json.loads(raw)["score"])
            except Exception:
                pass
        return None

    if not is_slm_available():
        _save_to_cache(prompt_hash, "judge", None, "offline_skip")
        return None

    try:
        raw = _call_slm(prompt)
        _save_to_cache(prompt_hash, "judge", raw, "cache_miss")
        return int(json.loads(raw)["score"])
    except Exception as exc:
        logger.warning("Erro ao avaliar coerência: %s", exc)
        return None
