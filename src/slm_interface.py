"""
SLM Interface Module
Handles communication with Docker-hosted Small Language Models (SLMs)
Supports Qwen2.5 and IBM Granite 4.0 Nano via HTTP API
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLMInterface:
    """Interface para comunicação com SLMs locais via Docker Model Runner"""
    
    def __init__(
        self,
        primary_url: str = "http://localhost:8080",
        backup_url: str = "http://localhost:8081",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize SLM interface
        
        Args:
            primary_url: URL for primary SLM (Qwen2.5)
            backup_url: URL for backup SLM (Granite 4.0 Nano)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.primary_url = primary_url
        self.backup_url = backup_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.current_url = primary_url
        self.cache = {}  # Simple cache for identical requests
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to SLM servers"""
        try:
            response = requests.get(
                f"{self.primary_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"✓ Connected to primary SLM at {self.primary_url}")
                return True
        except Exception as e:
            logger.warning(f"Primary SLM not available: {e}")
            
        try:
            response = requests.get(
                f"{self.backup_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"✓ Connected to backup SLM at {self.backup_url}")
                self.current_url = self.backup_url
                return True
        except Exception as e:
            logger.warning(f"Backup SLM not available: {e}")
        
        logger.error("No SLM servers available. Please run docker_setup.ps1")
        return False
    
    def generate_explanation(
        self,
        feature_importances: Dict[str, float],
        prediction: float,
        instance_features: Dict[str, any],
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> str:
        """
        Generate natural language explanation from feature importances
        
        Args:
            feature_importances: Dictionary of feature names to importance values
            prediction: Model prediction value
            instance_features: Dictionary of feature names to values for this instance
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Natural language explanation string
        """
        # Create cache key
        cache_key = (
            tuple(sorted(feature_importances.items())),
            prediction,
            tuple(sorted(instance_features.items()))
        )
        
        if cache_key in self.cache:
            logger.info("Returning cached explanation")
            return self.cache[cache_key]
        
        # Build prompt
        prompt = self._build_explanation_prompt(
            feature_importances,
            prediction,
            instance_features
        )
        
        # Generate explanation
        explanation = self._call_slm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Cache result
        self.cache[cache_key] = explanation
        
        return explanation
    
    def _build_explanation_prompt(
        self,
        feature_importances: Dict[str, float],
        prediction: float,
        instance_features: Dict[str, any]
    ) -> str:
        """Build prompt for explanation generation"""
        
        # Sort features by absolute importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        prompt = f"""You are an expert in explaining machine learning predictions for credit risk assessment.

A loan application was analyzed and the model predicted: {'DEFAULT (High Risk)' if prediction >= 0.5 else 'NO DEFAULT (Low Risk)'} with confidence {abs(prediction - 0.5) * 200:.1f}%.

Instance characteristics:
"""
        
        # Add instance features
        for feature, value in instance_features.items():
            prompt += f"- {feature}: {value}\n"
        
        prompt += f"\nFeature importance scores (LIME/SHAP):\n"
        
        # Add top features
        for feature, importance in sorted_features[:8]:
            direction = "increases" if importance > 0 else "decreases"
            prompt += f"- {feature}: {importance:.4f} ({direction} risk)\n"
        
        prompt += """\nProvide a clear, concise explanation in 2-3 sentences explaining:
1. Why the model made this prediction
2. Which factors were most influential
3. The practical meaning for the applicant

Explanation:"""
        
        return prompt
    
    def evaluate_explanation_coherence(
        self,
        explanation: str,
        max_tokens: int = 50
    ) -> float:
        """
        Ask SLM to self-evaluate explanation quality
        
        Args:
            explanation: The explanation text to evaluate
            max_tokens: Maximum tokens in response
            
        Returns:
            Quality score from 0-10
        """
        prompt = f"""Rate the following explanation for clarity, coherence, and usefulness on a scale of 0-10.
Only respond with a number between 0 and 10.

Explanation: "{explanation}"

Rating (0-10):"""
        
        response = self._call_slm(prompt, max_tokens=max_tokens, temperature=0.1)
        
        # Extract numeric rating
        try:
            # Try to find a number in the response
            import re
            match = re.search(r'\b(\d+(?:\.\d+)?)\b', response)
            if match:
                rating = float(match.group(1))
                return min(max(rating, 0), 10)  # Clamp to 0-10
        except Exception as e:
            logger.warning(f"Failed to parse rating: {e}")
        
        return 5.0  # Default middle rating
    
    def aggregate_explanations(
        self,
        explanations: List[str],
        aggregation_strategy: str = "synthesis",
        max_tokens: int = 300
    ) -> str:
        """
        Aggregate multiple weak explanations into a strong one
        
        Args:
            explanations: List of explanation texts
            aggregation_strategy: 'concatenation' or 'synthesis'
            max_tokens: Maximum tokens in response
            
        Returns:
            Aggregated explanation
        """
        if aggregation_strategy == "concatenation":
            prompt = f"""Below are {len(explanations)} explanations for the same prediction. 
Combine them into one clear, coherent explanation that captures the key insights.

Explanations:
"""
            for i, exp in enumerate(explanations, 1):
                prompt += f"\n{i}. {exp}\n"
            
            prompt += "\nSynthesized explanation:"
            
        else:  # synthesis strategy
            prompt = f"""Below are {len(explanations)} explanations for the same prediction.
Each explanation may emphasize different factors or perspectives.

Your task:
1. Identify the consensus view (factors mentioned most frequently)
2. Note any conflicting interpretations
3. Create a unified explanation that captures the most reliable insights

Explanations:
"""
            for i, exp in enumerate(explanations, 1):
                prompt += f"\n{i}. {exp}\n"
            
            prompt += "\nUnified explanation (2-3 sentences):"
        
        return self._call_slm(prompt, max_tokens=max_tokens, temperature=0.5)
    
    def _call_slm(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> str:
        """
        Make API call to SLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["\n\n\n"]
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.current_url}/v1/completions",
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if "choices" in result:
                        return result["choices"][0]["text"].strip()
                    elif "text" in result:
                        return result["text"].strip()
                    elif "completion" in result:
                        return result["completion"].strip()
                    else:
                        logger.warning(f"Unexpected response format: {result}")
                        return str(result)
                
                else:
                    logger.warning(f"API returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                time.sleep(1)
        
        # Fallback: return placeholder
        logger.error("All retry attempts failed")
        return "[SLM unavailable - explanation could not be generated]"
    
    def clear_cache(self):
        """Clear the explanation cache"""
        self.cache.clear()
        logger.info("Cache cleared")
