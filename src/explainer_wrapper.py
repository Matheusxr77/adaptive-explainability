"""
Explainer Wrapper Module
Provides unified interface for LIME and SHAP explainers with configurable perturbations
"""

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import shap
from typing import Dict, List, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainerWrapper:
    """Wrapper for LIME and SHAP explainers with tracking capabilities"""
    
    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        feature_names: List[str],
        categorical_features: List[str] = None,
        mode: str = "classification"
    ):
        """
        Initialize explainer wrapper
        
        Args:
            model: Trained model with predict_proba method
            X_train: Training data for background distribution
            feature_names: List of feature names
            categorical_features: List of categorical feature names
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.mode = mode
        
        # Initialize explainers
        self._init_lime()
        self._init_shap()
        
        # Tracking
        self.inference_count = 0
        self.total_time = 0
    
    def _init_lime(self):
        """Initialize LIME explainer"""
        categorical_indices = [
            i for i, name in enumerate(self.feature_names)
            if name in self.categorical_features
        ]
        
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            categorical_features=categorical_indices,
            mode=self.mode,
            random_state=42
        )
        logger.info("✓ LIME explainer initialized")
    
    def _init_shap(self):
        """Initialize SHAP explainer"""
        # Use TreeExplainer for tree-based models
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            logger.info("✓ SHAP TreeExplainer initialized")
        except Exception as e:
            # Fallback to KernelExplainer
            logger.warning(f"TreeExplainer failed: {e}. Using KernelExplainer.")
            background = shap.sample(self.X_train, 100)
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
            logger.info("✓ SHAP KernelExplainer initialized")
    
    def explain_lime(
        self,
        instance: pd.Series,
        num_samples: int = 100,
        top_k: int = 10
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Generate LIME explanation
        
        Args:
            instance: Instance to explain
            num_samples: Number of perturbations
            top_k: Number of top features to return
            
        Returns:
            (feature_importances, metadata)
        """
        start_time = time.time()
        
        # Track model calls
        initial_count = self.inference_count
        
        # Generate explanation
        exp = self.lime_explainer.explain_instance(
            data_row=instance.values,
            predict_fn=self._tracked_predict,
            num_samples=num_samples,
            num_features=top_k
        )
        
        # Extract feature importances
        importance_map = dict(exp.as_list())
        
        # Convert to feature name -> importance mapping
        feature_importances = {}
        for feature_desc, importance in importance_map.items():
            # Extract feature name (LIME returns descriptions like "age <= 5")
            feature_name = feature_desc.split()[0].split('<=')[0].split('>')[0].split('=')[0].strip()
            if feature_name in self.feature_names:
                feature_importances[feature_name] = importance
        
        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        
        metadata = {
            'method': 'LIME',
            'num_samples': num_samples,
            'inference_count': self.inference_count - initial_count,
            'time_seconds': elapsed_time,
            'feature_count': len(feature_importances)
        }
        
        return feature_importances, metadata
    
    def explain_shap(
        self,
        instance: pd.Series,
        top_k: int = 10
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Generate SHAP explanation
        
        Args:
            instance: Instance to explain
            top_k: Number of top features to return
            
        Returns:
            (feature_importances, metadata)
        """
        start_time = time.time()
        
        # Generate SHAP values
        shap_values = self.shap_explainer.shap_values(instance.values.reshape(1, -1))
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification: use positive class
            shap_values = shap_values[1]
        
        # Flatten if needed
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create feature importance dictionary
        feature_importances = {
            name: float(value)
            for name, value in zip(self.feature_names, shap_values)
        }
        
        # Sort by absolute importance and keep top_k
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        feature_importances = dict(sorted_features)
        
        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        
        metadata = {
            'method': 'SHAP',
            'inference_count': 1,  # SHAP TreeExplainer is efficient
            'time_seconds': elapsed_time,
            'feature_count': len(feature_importances)
        }
        
        return feature_importances, metadata
    
    def _tracked_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict function that tracks inference count"""
        self.inference_count += len(X)
        return self.model.predict_proba(X)
    
    def reset_tracking(self):
        """Reset inference tracking counters"""
        self.inference_count = 0
        self.total_time = 0
    
    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'total_inferences': self.inference_count,
            'total_time': self.total_time,
            'avg_time_per_inference': self.total_time / max(self.inference_count, 1)
        }


class ExplanationStabilityAnalyzer:
    """Analyze stability of explanations across multiple runs"""
    
    @staticmethod
    def compute_stability(
        explanations: List[Dict[str, float]]
    ) -> float:
        """
        Compute stability score based on variance in feature importances
        
        Args:
            explanations: List of feature importance dictionaries
            
        Returns:
            Stability score (lower variance = higher stability)
        """
        if len(explanations) < 2:
            return 1.0
        
        # Get all features
        all_features = set()
        for exp in explanations:
            all_features.update(exp.keys())
        
        # Compute variance for each feature
        variances = []
        for feature in all_features:
            values = [exp.get(feature, 0) for exp in explanations]
            variances.append(np.var(values))
        
        # Return mean variance (inverted so higher = more stable)
        mean_variance = np.mean(variances)
        
        # Convert to 0-1 scale where 1 = perfectly stable
        stability = 1 / (1 + mean_variance)
        
        return stability
    
    @staticmethod
    def compute_feature_agreement(
        exp1: Dict[str, float],
        exp2: Dict[str, float],
        top_k: int = 5
    ) -> float:
        """
        Compute Jaccard similarity of top-k features
        
        Args:
            exp1: First explanation
            exp2: Second explanation
            top_k: Number of top features to compare
            
        Returns:
            Jaccard similarity (0-1)
        """
        # Get top-k features
        top1 = set(sorted(exp1.keys(), key=lambda x: abs(exp1[x]), reverse=True)[:top_k])
        top2 = set(sorted(exp2.keys(), key=lambda x: abs(exp2[x]), reverse=True)[:top_k])
        
        # Compute Jaccard similarity
        intersection = len(top1 & top2)
        union = len(top1 | top2)
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def compute_directional_agreement(
        exp1: Dict[str, float],
        exp2: Dict[str, float]
    ) -> float:
        """
        Compute agreement in direction (sign) of feature importances
        
        Args:
            exp1: First explanation
            exp2: Second explanation
            
        Returns:
            Proportion of features with same sign (0-1)
        """
        common_features = set(exp1.keys()) & set(exp2.keys())
        
        if not common_features:
            return 0
        
        agreements = sum(
            1 for f in common_features
            if np.sign(exp1[f]) == np.sign(exp2[f])
        )
        
        return agreements / len(common_features)
