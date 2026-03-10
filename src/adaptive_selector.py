"""
Adaptive Perturbation Selector Module
Implements adaptive selection of perturbation count for LIME based on explanation coherence
Addresses Task 1.1: Adaptive selection of perturbations for coherent explanations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from .explainer_wrapper import ExplainerWrapper
from .slm_interface import SLMInterface
from .metrics import CoherenceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptivePerturbationSelector:
    """Selects optimal number of perturbations adaptively for each instance"""
    
    def __init__(
        self,
        explainer: ExplainerWrapper,
        slm_interface: SLMInterface,
        perturbation_levels: List[int] = None,
        coherence_threshold: float = 7.0,
        variance_threshold: float = 0.15,
        stability_runs: int = 3
    ):
        """
        Initialize adaptive selector
        
        Args:
            explainer: Explainer wrapper instance
            slm_interface: SLM interface for generating text explanations
            perturbation_levels: List of perturbation counts to test
            coherence_threshold: Minimum coherence score required
            variance_threshold: Maximum acceptable variance in feature importances
            stability_runs: Number of runs to test stability
        """
        self.explainer = explainer
        self.slm_interface = slm_interface
        self.perturbation_levels = perturbation_levels or [5, 10, 25, 50, 100, 250, 500, 1000]
        self.coherence_threshold = coherence_threshold
        self.variance_threshold = variance_threshold
        self.stability_runs = stability_runs
        
        # Results storage
        self.selection_history = []
    
    def select_optimal_perturbations(
        self,
        instance: pd.Series,
        prediction: float,
        instance_features: Dict[str, any],
        search_strategy: str = "binary"
    ) -> Tuple[int, Dict]:
        """
        Select optimal perturbation count for a given instance
        
        Args:
            instance: Instance to explain
            prediction: Model prediction for this instance
            instance_features: Feature values as dictionary
            search_strategy: 'binary' or 'sequential' search
            
        Returns:
            (optimal_perturbations, results_dict)
        """
        if search_strategy == "binary":
            return self._binary_search(instance, prediction, instance_features)
        else:
            return self._sequential_search(instance, prediction, instance_features)
    
    def _sequential_search(
        self,
        instance: pd.Series,
        prediction: float,
        instance_features: Dict[str, any]
    ) -> Tuple[int, Dict]:
        """Sequential search from lowest to highest perturbations"""
        
        results = {
            'tested_levels': [],
            'coherence_scores': [],
            'slm_ratings': [],
            'stability_variances': [],
            'inference_counts': [],
            'times': []
        }
        
        for num_perturbations in self.perturbation_levels:
            logger.info(f"Testing {num_perturbations} perturbations...")
            
            # Generate multiple explanations for stability
            explanations = []
            total_inferences = 0
            total_time = 0
            
            for run in range(self.stability_runs):
                self.explainer.reset_tracking()
                
                feature_importances, metadata = self.explainer.explain_lime(
                    instance,
                    num_samples=num_perturbations,
                    top_k=10
                )
                
                explanations.append(feature_importances)
                total_inferences += metadata['inference_count']
                total_time += metadata['time_seconds']
            
            # Generate text explanation from first run
            text_explanation = self.slm_interface.generate_explanation(
                explanations[0],
                prediction,
                instance_features
            )
            
            # Evaluate coherence
            slm_rating = self.slm_interface.evaluate_explanation_coherence(text_explanation)
            stability_variance = CoherenceMetrics.compute_feature_stability_variance(explanations)
            coherence_score = CoherenceMetrics.compute_composite_coherence_score(
                slm_rating,
                stability_variance,
                self.variance_threshold
            )
            
            # Store results
            results['tested_levels'].append(num_perturbations)
            results['coherence_scores'].append(coherence_score)
            results['slm_ratings'].append(slm_rating)
            results['stability_variances'].append(stability_variance)
            results['inference_counts'].append(total_inferences)
            results['times'].append(total_time)
            
            logger.info(f"  Coherence: {coherence_score:.2f}, SLM Rating: {slm_rating:.2f}, Variance: {stability_variance:.4f}")
            
            # Check if threshold met
            if coherence_score >= self.coherence_threshold:
                logger.info(f"✓ Threshold met at {num_perturbations} perturbations")
                results['selected_perturbations'] = num_perturbations
                results['explanation'] = text_explanation
                results['feature_importances'] = explanations[0]
                
                self.selection_history.append(results)
                return num_perturbations, results
        
        # If no level met threshold, return highest tested
        logger.warning(f"Threshold not met. Using maximum: {self.perturbation_levels[-1]}")
        results['selected_perturbations'] = self.perturbation_levels[-1]
        results['explanation'] = text_explanation
        results['feature_importances'] = explanations[0]
        
        self.selection_history.append(results)
        return self.perturbation_levels[-1], results
    
    def _binary_search(
        self,
        instance: pd.Series,
        prediction: float,
        instance_features: Dict[str, any]
    ) -> Tuple[int, Dict]:
        """Binary search for optimal perturbations (faster but less precise)"""
        
        # For binary search, we test fewer levels
        left = 0
        right = len(self.perturbation_levels) - 1
        best_level = self.perturbation_levels[-1]
        
        results = {
            'tested_levels': [],
            'coherence_scores': [],
            'slm_ratings': [],
            'stability_variances': [],
            'inference_counts': [],
            'times': []
        }
        
        while left <= right:
            mid = (left + right) // 2
            num_perturbations = self.perturbation_levels[mid]
            
            logger.info(f"Testing {num_perturbations} perturbations (binary search)...")
            
            # Test this level
            explanations = []
            total_inferences = 0
            total_time = 0
            
            for run in range(self.stability_runs):
                self.explainer.reset_tracking()
                
                feature_importances, metadata = self.explainer.explain_lime(
                    instance,
                    num_samples=num_perturbations,
                    top_k=10
                )
                
                explanations.append(feature_importances)
                total_inferences += metadata['inference_count']
                total_time += metadata['time_seconds']
            
            # Generate and evaluate explanation
            text_explanation = self.slm_interface.generate_explanation(
                explanations[0],
                prediction,
                instance_features
            )
            
            slm_rating = self.slm_interface.evaluate_explanation_coherence(text_explanation)
            stability_variance = CoherenceMetrics.compute_feature_stability_variance(explanations)
            coherence_score = CoherenceMetrics.compute_composite_coherence_score(
                slm_rating,
                stability_variance,
                self.variance_threshold
            )
            
            # Store results
            results['tested_levels'].append(num_perturbations)
            results['coherence_scores'].append(coherence_score)
            results['slm_ratings'].append(slm_rating)
            results['stability_variances'].append(stability_variance)
            results['inference_counts'].append(total_inferences)
            results['times'].append(total_time)
            
            logger.info(f"  Coherence: {coherence_score:.2f}")
            
            if coherence_score >= self.coherence_threshold:
                # Threshold met, try lower
                best_level = num_perturbations
                right = mid - 1
            else:
                # Need more perturbations
                left = mid + 1
        
        results['selected_perturbations'] = best_level
        results['explanation'] = text_explanation
        results['feature_importances'] = explanations[0]
        
        self.selection_history.append(results)
        return best_level, results
    
    def analyze_instance_characteristics(
        self,
        instance: pd.Series,
        model,
        X_train: pd.DataFrame
    ) -> Dict:
        """
        Analyze characteristics of an instance that might affect optimal perturbations
        
        Args:
            instance: Instance to analyze
            model: Trained model
            X_train: Training data
            
        Returns:
            Dictionary of characteristics
        """
        # Get prediction probability
        pred_proba = model.predict_proba(instance.values.reshape(1, -1))[0]
        confidence = max(pred_proba)
        
        # Distance to decision boundary (certainty)
        boundary_distance = abs(max(pred_proba) - 0.5)
        
        # Distance to nearest training samples (novelty)
        from scipy.spatial.distance import cdist
        distances = cdist(
            instance.values.reshape(1, -1),
            X_train.values,
            metric='euclidean'
        )[0]
        nearest_distance = np.min(distances)
        avg_distance = np.mean(distances)
        
        # Feature value ranges (complexity)
        feature_std = np.std(instance.values)
        
        return {
            'confidence': confidence,
            'boundary_distance': boundary_distance,
            'nearest_neighbor_distance': nearest_distance,
            'avg_training_distance': avg_distance,
            'feature_std': feature_std,
            'is_certain': boundary_distance > 0.3,
            'is_novel': nearest_distance > avg_distance
        }
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of adaptive selection results"""
        if not self.selection_history:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.selection_history:
            summary_data.append({
                'selected_perturbations': result['selected_perturbations'],
                'final_coherence': result['coherence_scores'][-1],
                'final_slm_rating': result['slm_ratings'][-1],
                'final_variance': result['stability_variances'][-1],
                'levels_tested': len(result['tested_levels']),
                'total_inferences': sum(result['inference_counts']),
                'total_time': sum(result['times'])
            })
        
        return pd.DataFrame(summary_data)
