"""
Adaptive Explainability Package
Implements adaptive perturbation selection and explanation aggregation for LIME/SHAP
"""

from .model_trainer import CreditRiskModelTrainer
from .slm_interface import SLMInterface
from .explainer_wrapper import ExplainerWrapper, ExplanationStabilityAnalyzer
from .adaptive_selector import AdaptivePerturbationSelector
from .explanation_aggregator import ExplanationAggregator
from .metrics import CoherenceMetrics, ComputationalCostMetrics

__all__ = [
    'CreditRiskModelTrainer',
    'SLMInterface',
    'ExplainerWrapper',
    'ExplanationStabilityAnalyzer',
    'AdaptivePerturbationSelector',
    'ExplanationAggregator',
    'CoherenceMetrics',
    'ComputationalCostMetrics'
]

__version__ = '1.0.0'
