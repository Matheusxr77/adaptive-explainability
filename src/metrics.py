"""
Metrics Module
Evaluation metrics for explanation quality and coherence
"""

import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoherenceMetrics:
    """Metrics for evaluating explanation coherence and quality"""
    
    @staticmethod
    def compute_feature_stability_variance(
        explanations: List[Dict[str, float]]
    ) -> float:
        """
        Compute variance in feature importances across multiple explanations
        Lower variance indicates more stable/coherent explanations
        
        Args:
            explanations: List of feature importance dictionaries
            
        Returns:
            Mean variance across features
        """
        if len(explanations) < 2:
            return 0.0
        
        # Collect all features
        all_features = set()
        for exp in explanations:
            all_features.update(exp.keys())
        
        # Compute variance for each feature
        variances = []
        for feature in all_features:
            values = [exp.get(feature, 0) for exp in explanations]
            if len(values) > 1:
                variances.append(np.var(values))
        
        return np.mean(variances) if variances else 0.0
    
    @staticmethod
    def compute_composite_coherence_score(
        slm_rating: float,
        stability_variance: float,
        variance_threshold: float = 0.15
    ) -> float:
        """
        Compute composite coherence score combining SLM rating and stability
        
        Args:
            slm_rating: Rating from SLM (0-10)
            stability_variance: Variance in feature importances
            variance_threshold: Threshold for acceptable variance
            
        Returns:
            Composite score (0-10)
        """
        # Normalize SLM rating to 0-1
        slm_component = slm_rating / 10.0
        
        # Stability component (1 if variance below threshold, scaled down if higher)
        if stability_variance <= variance_threshold:
            stability_component = 1.0
        else:
            stability_component = variance_threshold / stability_variance
            stability_component = min(stability_component, 1.0)
        
        # Weighted combination (60% SLM, 40% stability)
        composite = 0.6 * slm_component + 0.4 * stability_component
        
        # Scale back to 0-10
        return composite * 10.0
    
    @staticmethod
    def compute_jaccard_similarity(
        set1: set,
        set2: set
    ) -> float:
        """
        Compute Jaccard similarity between two sets
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity (0-1)
        """
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_feature_alignment(
        ground_truth: Dict[str, float],
        comparison: Dict[str, float],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Compute alignment between two explanations
        
        Args:
            ground_truth: Reference explanation (e.g., high-perturbation LIME)
            comparison: Explanation to compare (e.g., aggregated weak explanations)
            top_k: Number of top features to consider
            
        Returns:
            Dictionary with alignment metrics
        """
        # Get top-k features from each
        gt_top = set(sorted(
            ground_truth.keys(),
            key=lambda x: abs(ground_truth[x]),
            reverse=True
        )[:top_k])
        
        comp_top = set(sorted(
            comparison.keys(),
            key=lambda x: abs(comparison[x]),
            reverse=True
        )[:top_k])
        
        # Jaccard similarity of top features
        jaccard = CoherenceMetrics.compute_jaccard_similarity(gt_top, comp_top)
        
        # Directional agreement (same sign) for common features
        common_features = gt_top & comp_top
        if common_features:
            directional_agreement = sum(
                1 for f in common_features
                if np.sign(ground_truth.get(f, 0)) == np.sign(comparison.get(f, 0))
            ) / len(common_features)
        else:
            directional_agreement = 0.0
        
        # Importance correlation for common features
        if len(common_features) >= 2:
            gt_values = [ground_truth.get(f, 0) for f in common_features]
            comp_values = [comparison.get(f, 0) for f in common_features]
            correlation = np.corrcoef(gt_values, comp_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'jaccard_similarity': jaccard,
            'directional_agreement': directional_agreement,
            'importance_correlation': correlation,
            'common_features_count': len(common_features)
        }
    
    @staticmethod
    def compute_explanation_diversity(
        explanations: List[Dict[str, float]]
    ) -> float:
        """
        Compute diversity across multiple explanations
        Higher diversity suggests less consensus
        
        Args:
            explanations: List of feature importance dictionaries
            
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(explanations) < 2:
            return 0.0
        
        # Compute pairwise Jaccard dissimilarity
        dissimilarities = []
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                set_i = set(explanations[i].keys())
                set_j = set(explanations[j].keys())
                jaccard = CoherenceMetrics.compute_jaccard_similarity(set_i, set_j)
                dissimilarities.append(1 - jaccard)
        
        return np.mean(dissimilarities)


class ComputationalCostMetrics:
    """Metrics for tracking computational cost"""
    
    @staticmethod
    def compute_cost_comparison(
        weak_explanation_costs: List[float],
        aggregation_cost: float,
        strong_explanation_cost: float
    ) -> Dict[str, float]:
        """
        Compare computational costs: N weak + aggregation vs 1 strong
        
        Args:
            weak_explanation_costs: List of costs for weak explanations
            aggregation_cost: Cost of aggregation step
            strong_explanation_cost: Cost of strong explanation
            
        Returns:
            Dictionary with cost metrics
        """
        total_weak_cost = sum(weak_explanation_costs) + aggregation_cost
        
        cost_ratio = total_weak_cost / strong_explanation_cost if strong_explanation_cost > 0 else 0
        cost_savings = strong_explanation_cost - total_weak_cost
        cost_savings_percent = (cost_savings / strong_explanation_cost * 100) if strong_explanation_cost > 0 else 0
        
        return {
            'weak_explanations_count': len(weak_explanation_costs),
            'total_weak_cost': total_weak_cost,
            'strong_explanation_cost': strong_explanation_cost,
            'aggregation_cost': aggregation_cost,
            'cost_ratio': cost_ratio,
            'cost_savings': cost_savings,
            'cost_savings_percent': cost_savings_percent,
            'is_cheaper': total_weak_cost < strong_explanation_cost
        }
    
    @staticmethod
    def inferences_to_time(
        inference_count: int,
        avg_inference_time: float = 0.001
    ) -> float:
        """
        Estimate time from inference count
        
        Args:
            inference_count: Number of model inferences
            avg_inference_time: Average time per inference in seconds
            
        Returns:
            Estimated total time in seconds
        """
        return inference_count * avg_inference_time
