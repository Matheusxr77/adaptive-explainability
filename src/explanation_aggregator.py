"""
Explanation Aggregator Module
Aggregates multiple weak explanations into a strong unified explanation
Addresses Task 1.2: Combining N poor explanations into one strong explanation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from .slm_interface import SLMInterface
from .metrics import CoherenceMetrics, ComputationalCostMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationAggregator:
    """Aggregates multiple weak explanations into a strong one"""
    
    def __init__(
        self,
        slm_interface: SLMInterface
    ):
        """
        Initialize explanation aggregator
        
        Args:
            slm_interface: SLM interface for aggregation
        """
        self.slm_interface = slm_interface
        self.aggregation_history = []
    
    def aggregate_weak_explanations(
        self,
        weak_explanations: List[Dict],
        strategy: str = "synthesis",
        include_costs: bool = True
    ) -> Dict:
        """
        Aggregate multiple weak explanations
        
        Args:
            weak_explanations: List of dicts with 'text', 'feature_importances', 'cost'
            strategy: 'concatenation' or 'synthesis'
            include_costs: Whether to compute cost metrics
            
        Returns:
            Dictionary with aggregated explanation and metrics
        """
        logger.info(f"Aggregating {len(weak_explanations)} weak explanations using {strategy}")
        
        # Extract text explanations
        text_explanations = [exp['text'] for exp in weak_explanations]
        
        # Aggregate using SLM
        aggregated_text = self.slm_interface.aggregate_explanations(
            text_explanations,
            aggregation_strategy=strategy
        )
        
        # Aggregate feature importances (average)
        aggregated_features = self._aggregate_feature_importances(
            [exp['feature_importances'] for exp in weak_explanations]
        )
        
        # Evaluate coherence of aggregated explanation
        coherence_rating = self.slm_interface.evaluate_explanation_coherence(
            aggregated_text
        )
        
        result = {
            'aggregated_text': aggregated_text,
            'aggregated_features': aggregated_features,
            'coherence_rating': coherence_rating,
            'num_weak_explanations': len(weak_explanations),
            'strategy': strategy
        }
        
        # Compute costs if requested
        if include_costs:
            weak_costs = [exp.get('cost', 0) for exp in weak_explanations]
            aggregation_cost = 1  # Simplified: 1 SLM call for aggregation
            
            result['weak_costs'] = weak_costs
            result['aggregation_cost'] = aggregation_cost
            result['total_cost'] = sum(weak_costs) + aggregation_cost
        
        self.aggregation_history.append(result)
        
        return result
    
    def _aggregate_feature_importances(
        self,
        importance_lists: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate feature importances by averaging
        
        Args:
            importance_lists: List of feature importance dictionaries
            
        Returns:
            Averaged feature importances
        """
        # Collect all features
        all_features = set()
        for imp_dict in importance_lists:
            all_features.update(imp_dict.keys())
        
        # Average each feature
        aggregated = {}
        for feature in all_features:
            values = [imp_dict.get(feature, 0) for imp_dict in importance_lists]
            aggregated[feature] = np.mean(values)
        
        return aggregated
    
    def compare_with_ground_truth(
        self,
        aggregated_result: Dict,
        ground_truth: Dict,
        strong_explanation_cost: float
    ) -> Dict:
        """
        Compare aggregated explanation with ground truth (high-perturbation explanation)
        
        Args:
            aggregated_result: Result from aggregate_weak_explanations
            ground_truth: Dict with 'text', 'feature_importances', 'cost'
            strong_explanation_cost: Cost of generating strong explanation
            
        Returns:
            Comparison metrics
        """
        # Feature alignment
        alignment_metrics = CoherenceMetrics.compute_feature_alignment(
            ground_truth['feature_importances'],
            aggregated_result['aggregated_features'],
            top_k=5
        )
        
        # Coherence comparison
        gt_coherence = self.slm_interface.evaluate_explanation_coherence(
            ground_truth['text']
        )
        agg_coherence = aggregated_result['coherence_rating']
        
        coherence_diff = agg_coherence - gt_coherence
        coherence_ratio = agg_coherence / gt_coherence if gt_coherence > 0 else 0
        
        # Cost comparison
        cost_metrics = ComputationalCostMetrics.compute_cost_comparison(
            aggregated_result.get('weak_costs', []),
            aggregated_result.get('aggregation_cost', 0),
            strong_explanation_cost
        )
        
        comparison = {
            'alignment_metrics': alignment_metrics,
            'ground_truth_coherence': gt_coherence,
            'aggregated_coherence': agg_coherence,
            'coherence_difference': coherence_diff,
            'coherence_ratio': coherence_ratio,
            'cost_metrics': cost_metrics,
            'quality_per_cost': agg_coherence / cost_metrics['total_weak_cost'] if cost_metrics['total_weak_cost'] > 0 else 0
        }
        
        # Log summary
        logger.info(f"\n=== Aggregation vs Ground Truth Comparison ===")
        logger.info(f"Feature Alignment (Jaccard): {alignment_metrics['jaccard_similarity']:.3f}")
        logger.info(f"Directional Agreement: {alignment_metrics['directional_agreement']:.3f}")
        logger.info(f"Coherence Ratio: {coherence_ratio:.3f}")
        logger.info(f"Cost Savings: {cost_metrics['cost_savings_percent']:.1f}%")
        logger.info(f"Quality per Cost: {comparison['quality_per_cost']:.3f}")
        
        return comparison
    
    def batch_aggregate_and_compare(
        self,
        instances_data: List[Dict],
        strategies: List[str] = None
    ) -> pd.DataFrame:
        """
        Run aggregation and comparison for multiple instances
        
        Args:
            instances_data: List of dicts with 'weak_explanations' and 'ground_truth'
            strategies: List of aggregation strategies to test
            
        Returns:
            DataFrame with results for all instances and strategies
        """
        if strategies is None:
            strategies = ['concatenation', 'synthesis']
        
        results = []
        
        for idx, instance_data in enumerate(instances_data):
            logger.info(f"\nProcessing instance {idx + 1}/{len(instances_data)}")
            
            for strategy in strategies:
                # Aggregate
                agg_result = self.aggregate_weak_explanations(
                    instance_data['weak_explanations'],
                    strategy=strategy
                )
                
                # Compare with ground truth
                comparison = self.compare_with_ground_truth(
                    agg_result,
                    instance_data['ground_truth'],
                    instance_data['ground_truth'].get('cost', 0)
                )
                
                # Compile results
                result_row = {
                    'instance_id': idx,
                    'strategy': strategy,
                    'num_weak': len(instance_data['weak_explanations']),
                    'jaccard_similarity': comparison['alignment_metrics']['jaccard_similarity'],
                    'directional_agreement': comparison['alignment_metrics']['directional_agreement'],
                    'importance_correlation': comparison['alignment_metrics']['importance_correlation'],
                    'coherence_ratio': comparison['coherence_ratio'],
                    'cost_ratio': comparison['cost_metrics']['cost_ratio'],
                    'cost_savings_percent': comparison['cost_metrics']['cost_savings_percent'],
                    'is_cheaper': comparison['cost_metrics']['is_cheaper'],
                    'quality_per_cost': comparison['quality_per_cost']
                }
                
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def find_optimal_weak_count(
        self,
        instance_data: Dict,
        weak_counts: List[int] = None,
        strategy: str = "synthesis"
    ) -> Tuple[int, pd.DataFrame]:
        """
        Find optimal number of weak explanations to aggregate
        
        Args:
            instance_data: Dict with all weak explanations and ground truth
            weak_counts: List of N values to test (e.g., [5, 10, 20, 50])
            strategy: Aggregation strategy
            
        Returns:
            (optimal_n, results_dataframe)
        """
        if weak_counts is None:
            weak_counts = [5, 10, 20, 50]
        
        all_weak = instance_data['weak_explanations']
        ground_truth = instance_data['ground_truth']
        
        results = []
        
        for n in weak_counts:
            if n > len(all_weak):
                logger.warning(f"Requested {n} weak explanations but only {len(all_weak)} available")
                continue
            
            # Sample n weak explanations
            sampled_weak = all_weak[:n]
            
            # Aggregate
            agg_result = self.aggregate_weak_explanations(
                sampled_weak,
                strategy=strategy
            )
            
            # Compare
            comparison = self.compare_with_ground_truth(
                agg_result,
                ground_truth,
                ground_truth.get('cost', 0)
            )
            
            results.append({
                'n': n,
                'jaccard_similarity': comparison['alignment_metrics']['jaccard_similarity'],
                'directional_agreement': comparison['alignment_metrics']['directional_agreement'],
                'coherence_ratio': comparison['coherence_ratio'],
                'cost_ratio': comparison['cost_metrics']['cost_ratio'],
                'quality_per_cost': comparison['quality_per_cost']
            })
        
        df_results = pd.DataFrame(results)
        
        # Find optimal based on quality_per_cost
        optimal_idx = df_results['quality_per_cost'].idxmax()
        optimal_n = df_results.loc[optimal_idx, 'n']
        
        logger.info(f"\nOptimal number of weak explanations: {optimal_n}")
        
        return int(optimal_n), df_results
