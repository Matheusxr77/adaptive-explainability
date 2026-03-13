"""
01_modelo_baseline.py — Treina o RandomForestClassifier e serializa artefatos.
Exige AUC-ROC >= 0.80. Salva CSVs e gráficos do baseline.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_PATH, TARGET_COL, NUMERICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS,
    RANDOM_STATE, TEST_SIZE, MIN_AUC_ROC,
    MODELS_DIR, RF_MODEL_PATH, PREPROCESSOR_PATH,
    BASELINE_METRICS_CSV, BASELINE_PREDICTIONS_CSV, BASELINE_IMPORTANCE_CSV,
    FIG_DIR,
)
from src.io_utils import save_csv, save_fig
import src.plotting as P

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor(feature_names_out: bool = False):
    """Constrói o ColumnTransformer de pré-processamento."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["A", "B", "C", "D", "E", "F", "G"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("ord", ord_pipeline, ORDINAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS),
    ], remainder="drop")


def get_feature_names(preprocessor, cat_cols: list[str]) -> list[str]:
    """Recupera os nomes das features após transformação."""
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
    return NUMERICAL_COLS + ORDINAL_COLS + ohe_names


def run_baseline() -> tuple:
    """
    Treina o modelo baseline.
    Retorna (model, preprocessor, X_test_transformed, y_test, feature_names).
    """
    logger.info("=== Baseline | Início ===")

    # ── Dados ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)]

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Treino: %d | Teste: %d", len(X_train), len(X_test))

    # ── Pré-processamento ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, CATEGORICAL_COLS)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train)

    # ── Avaliação ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)

    logger.info("AUC-ROC: %.4f | F1-macro: %.4f | Accuracy: %.4f | AP: %.4f",
                auc, f1_mac, acc, ap)

    if auc < MIN_AUC_ROC:
        raise RuntimeError(f"AUC-ROC = {auc:.4f} < mínimo exigido ({MIN_AUC_ROC})")

    print(f"AUC-ROC: {auc:.4f} ✓")

    # ── Serialização ──────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RF_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Modelo e pré-processador salvos em models/")

    # ── CSVs ──────────────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame([{
        "auc_roc": auc, "f1_macro": f1_mac, "accuracy": acc, "avg_precision": ap,
    }])
    save_csv(metrics_df, BASELINE_METRICS_CSV)

    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba})
    save_csv(preds_df, BASELINE_PREDICTIONS_CSV)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    save_csv(importance_df, BASELINE_IMPORTANCE_CSV)

    # ── Gráficos ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig = P.confusion_matrix_plot(cm, ["Não inadim.", "Inadim."], "Matriz de Confusão — Baseline")
    save_fig(fig, FIG_DIR / "baseline_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = P.roc_curve_plot(fpr, tpr, auc)
    save_fig(fig, FIG_DIR / "baseline_roc_curve.png")

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig = P.precision_recall_curve_plot(prec, rec, ap)
    save_fig(fig, FIG_DIR / "baseline_precision_recall_curve.png")

    fig = P.feature_importance_bar(importance_df, top_n=15)
    save_fig(fig, FIG_DIR / "baseline_feature_importance_top15.png")

    fig = P.score_distribution(y_proba, y_test.values)
    save_fig(fig, FIG_DIR / "baseline_prediction_score_distribution.png")

    logger.info("=== Baseline | Concluído ===")
    return model, preprocessor, X_test_t, y_test, feature_names, X_train_t


if __name__ == "__main__":
    run_baseline()
