"""
config.py — Constantes globais e caminhos do projeto.
Todos os scripts importam daqui para garantir consistência.
"""

from pathlib import Path

# ── Raiz do projeto ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Dados ─────────────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
TARGET_COL = "loan_status"
ORDINAL_COLS = ["loan_grade"]
CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
NUMERICAL_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
]

# ── Reprodutibilidade ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Modelo ────────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MIN_AUC_ROC = 0.80

# ── Resultados ────────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "resultados"
CSV_DIR = RESULTS_DIR / "csv"
FIG_DIR = RESULTS_DIR / "figuras"
CACHE_DIR = RESULTS_DIR / "cache"
REPORT_DIR = ROOT / "relatorio"

# ── SLM ───────────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://localhost:12434/engines/llama.cpp/v1"
LLM_MODEL = "ai/qwen2.5"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 300
LLM_TOP_P = 0.9

# ── LIME ──────────────────────────────────────────────────────────────────────
LIME_BASE_N_SAMPLES = 500
LIME_N_PERTURBATIONS_GRID = [25, 50, 100, 200, 300, 500, 750, 1000]
LIME_REFERENCE_N = 1000
LIME_N_INSTANCES_Q11 = 30
LIME_REPETITIONS = 3
SPEARMAN_THRESHOLD = 0.85
TOP_K = 5

# ── Caminhos de artefatos ─────────────────────────────────────────────────────
EDA_MISSINGNESS_CSV = CSV_DIR / "eda_missingness.csv"
EDA_NUMERIC_CSV = CSV_DIR / "eda_numeric_summary.csv"
EDA_CATEGORICAL_CSV = CSV_DIR / "eda_categorical_summary.csv"
EDA_INSIGHTS_CSV = CSV_DIR / "eda_insights.csv"

BASELINE_METRICS_CSV = CSV_DIR / "baseline_metrics.csv"
BASELINE_PREDICTIONS_CSV = CSV_DIR / "baseline_predictions_test.csv"
BASELINE_IMPORTANCE_CSV = CSV_DIR / "baseline_feature_importance.csv"

LIME_BASE_INSTANCES_CSV = CSV_DIR / "lime_base_selected_instances.csv"
LIME_BASE_CONTRIBUTIONS_CSV = CSV_DIR / "lime_base_feature_contributions.csv"
LIME_BASE_TEXTS_CSV = CSV_DIR / "lime_base_text_explanations.csv"

Q11_RAW_CSV = CSV_DIR / "q11_raw_runs.csv"
Q11_INSTANCE_CSV = CSV_DIR / "q11_instance_level_summary.csv"
Q11_N_LEVEL_CSV = CSV_DIR / "q11_n_level_summary.csv"
Q11_ADAPTIVE_CSV = CSV_DIR / "q11_adaptive_n_by_instance.csv"
Q11_TEXTS_CSV = CSV_DIR / "q11_text_explanations.csv"
Q11_CONCLUSIONS_CSV = CSV_DIR / "q11_final_conclusions.csv"

LLM_PROMPTS_CSV = CACHE_DIR / "llm_prompts.csv"
LLM_RESPONSES_JSON = CACHE_DIR / "llm_responses.json"

REPORT_MD = REPORT_DIR / "relatorio_final.md"
REPORT_CONCLUSIONS_CSV = REPORT_DIR / "conclusoes_finais.csv"
REPORT_METRICS_CSV = REPORT_DIR / "metricas_resumo.csv"


def ensure_dirs() -> None:
    """Cria todos os diretórios obrigatórios caso não existam."""
    for d in [MODELS_DIR, CSV_DIR, FIG_DIR, CACHE_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
