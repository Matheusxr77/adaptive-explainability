"""
main.py — Pipeline principal do projeto.
Executa todas as etapas em sequência.
Uso: python main.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def banner(msg: str) -> None:
    """Mostra uma faixa visual no log para separar melhor cada etapa."""
    sep = "=" * 60
    logger.info(sep)
    logger.info(msg)
    logger.info(sep)


def run_step(label: str, fn: Callable[[], Any]) -> Any:
    """Executa uma etapa do pipeline e registra tempo, sucesso ou falha."""
    banner(f"ETAPA: {label}")
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        logger.info("✓ %s concluído em %.1fs", label, elapsed)
        return result
    except Exception as exc:
        logger.error("✗ FALHA em %s: %s", label, exc, exc_info=True)
        raise


def main() -> None:
    """Orquestra o pipeline completo, do EDA ao relatório final."""
    banner("PIPELINE PRINCIPAL — Explicabilidade Adaptativa Q1.1")

    # Garante que a estrutura mínima de pastas exista antes de salvar artefatos.
    ensure_dirs()

    import importlib

    # O carregamento dinâmico evita problemas com nomes de módulos que começam
    # com números, como `00_eda.py` e `01_modelo_baseline.py`.
    eda_mod = importlib.import_module("scripts.00_eda")
    baseline_mod = importlib.import_module("scripts.01_modelo_baseline")
    lime_base_mod = importlib.import_module("scripts.02_lime_base")
    q11_mod = importlib.import_module("scripts.03_questao_1_1")
    report_mod = importlib.import_module("scripts.04_relatorio_final")

    # A ordem abaixo segue o fluxo natural do projeto:
    # primeiro entender os dados, depois treinar, explicar, experimentar e,
    # por fim, consolidar tudo no relatório.
    run_step("00 | Análise Exploratória", eda_mod.run_eda)
    run_step("01 | Modelo Baseline", baseline_mod.run_baseline)
    run_step("02 | LIME Base", lime_base_mod.run_lime_base)
    run_step("03 | Experimento Q1.1", q11_mod.run_q11)
    run_step("04 | Relatório Final", report_mod.run_report)

    banner("PIPELINE CONCLUÍDO")


if __name__ == "__main__":
    main()
