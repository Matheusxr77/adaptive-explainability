"""
io_utils.py — Helpers para leitura e escrita de artefatos em disco.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Salva DataFrame em CSV, criando diretórios intermediários."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")
    logger.info("CSV salvo: %s (%d linhas)", path, len(df))


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Carrega CSV com tratamento de erro descritivo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path, encoding="utf-8", **kwargs)
    logger.info("CSV carregado: %s (%d linhas)", path, len(df))
    return df


# ── JSON ──────────────────────────────────────────────────────────────────────

def save_json(data: dict | list, path: Path) -> None:
    """Salva estrutura como JSON com indentação."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("JSON salvo: %s", path)


def load_json(path: Path) -> dict | list:
    """Carrega JSON existente ou retorna estrutura vazia."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Figuras ───────────────────────────────────────────────────────────────────

def save_fig(fig, path: Path, *, dpi: int = 150) -> None:
    """Salva figura matplotlib em disco."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", path)


# ── Incremental CSV ───────────────────────────────────────────────────────────

def append_row_csv(row: dict, path: Path) -> None:
    """Adiciona uma linha ao CSV (cria com cabeçalho se não existir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")
