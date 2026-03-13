"""
data_loader.py — Helper compartilhado para recarregar dados processados
sem necessidade de re-treinar o pré-processador.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_PATH, TARGET_COL, NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from sklearn.model_selection import train_test_split


def load_processed_data(preprocessor):
    """
    Recarrega o dataset limpo, aplica o pré-processador já treinado
    e retorna (X_test_t, y_test, feature_names, X_train_t).
    """
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)]

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Nomes das features
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    ohe_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    feature_names = NUMERICAL_COLS + ORDINAL_COLS + ohe_names

    return X_test_t, y_test, feature_names, X_train_t
