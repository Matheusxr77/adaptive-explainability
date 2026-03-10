"""
Model Training Module
Handles data preprocessing, model training, and evaluation for credit risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Treina e avalia modelo XGBoost para previsão de risco de crédito"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def load_and_preprocess_data(
        self,
        data_path: str,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and preprocess credit risk data
        
        Args:
            data_path: Path to CSV file
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Display basic info
        logger.info(f"\nMissing values:\n{df.isnull().sum()}")
        logger.info(f"\nTarget distribution:\n{df['loan_status'].value_counts()}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Separate features and target
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        
        # Identify categorical and numerical features
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Categorical features: {self.categorical_features}")
        logger.info(f"Numerical features: {self.numerical_features}")
        
        # Encode categorical variables
        X = self._encode_categorical(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale numerical features
        X_train = self._scale_features(X_train, fit=True)
        X_test = self._scale_features(X_test, fit=False)
        
        self.feature_names = X_train.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset"""
        df = df.copy()
        
        # Fill numerical missing values with median
        numerical_cols = df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        X = X.copy()
        
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features"""
        X = X.copy()
        
        if fit:
            X[self.numerical_features] = self.scaler.fit_transform(X[self.numerical_features])
        else:
            X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        
        return X
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Dict = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            params: Model hyperparameters
            
        Returns:
            Trained model
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        
        logger.info("✓ Model training complete")
        return self.model
    
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Display results
        logger.info("\n=== Model Performance ===")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: str):
        """Save trained model and preprocessing objects"""
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and preprocessing objects"""
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.categorical_features = data['categorical_features']
        self.numerical_features = data['numerical_features']
        logger.info(f"Model loaded from {path}")
