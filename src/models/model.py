"""
Heart disease prediction model.
This module contains the model for predicting heart disease.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing.preprocessor import HeartDiseasePreprocessor

class HeartDiseaseModel:
    """
    Heart disease prediction model class.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the heart disease prediction model.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'svm', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = HeartDiseasePreprocessor()
        
    def train(self, X, y):
        """
        Train the model on the given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            self: The trained model
        """
        # Preprocess the data
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        # Create and train the model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.model.fit(X_preprocessed, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predicted classes
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess the data
        X_preprocessed = self.preprocessor.transform(X)
        
        # Make predictions
        return self.model.predict(X_preprocessed)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained model.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess the data
        X_preprocessed = self.preprocessor.transform(X)
        
        # Predict probabilities
        return self.model.predict_proba(X_preprocessed)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Make predictions
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob)
        }
        
        return metrics
    
    def save(self, model_path):
        """
        Save the model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Create a dictionary with all objects to save
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type
        }
        
        # Save the model data
        joblib.dump(model_data, model_path)
    
    @classmethod
    def load(cls, model_path):
        """
        Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            HeartDiseaseModel: The loaded model
        """
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Create a new instance
        instance = cls(model_type=model_data['model_type'])
        
        # Restore the model and preprocessor
        instance.model = model_data['model']
        instance.preprocessor = model_data['preprocessor']
        
        return instance
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not provide feature importances.")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a simplified feature importance dictionary
        # This avoids complex feature name extraction that might be version-dependent
        feature_importance = {}
        
        # Use original feature names for simplicity
        for i, feature in enumerate(self.preprocessor.numeric_features + self.preprocessor.categorical_features):
            if i < len(importances):
                feature_importance[feature] = importances[i]
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def explain_prediction(self, input_data):
        """
        Explain the prediction for a single input.
        
        Args:
            input_data (dict or pd.DataFrame): Input data
            
        Returns:
            dict: Explanation of the prediction
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = self.predict(input_data)[0]
        probability = self.predict_proba(input_data)[0][1]
        
        # Get risk level
        risk_level = self._get_risk_level(probability)
        
        # Get contributing factors
        contributing_factors = self._get_contributing_factors(input_data)
        
        # Create explanation
        explanation = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'contributing_factors': contributing_factors
        }
        
        return explanation
    
    def _get_risk_level(self, probability):
        """
        Get risk level based on probability.
        
        Args:
            probability (float): Probability of heart disease
            
        Returns:
            str: Risk level
        """
        if probability < 0.2:
            return "Low Risk"
        elif probability < 0.5:
            return "Moderate Risk"
        elif probability < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _get_contributing_factors(self, input_data, top_n=5):
        """
        Get top contributing factors for the prediction.
        
        Args:
            input_data (pd.DataFrame): Input data
            top_n (int): Number of top factors to return
            
        Returns:
            list: Top contributing factors
        """
        # Create a list to store factors
        factors = []
        
        # Get feature importance if available
        try:
            feature_importance = self.get_feature_importance()
        except:
            feature_importance = {}
            
            # If feature importance is not available, assign equal importance
            for feature in input_data.columns:
                feature_importance[feature] = 1.0 / len(input_data.columns)
        
        # Process each feature in the input data
        for feature, importance in feature_importance.items():
            if feature in input_data.columns:
                value = input_data[feature].iloc[0]
                
                # Get the description from feature definitions
                if feature in self.preprocessor.features:
                    feature_def = self.preprocessor.features[feature]
                    description = feature_def.get('description', feature)
                    
                    # Handle categorical features with value descriptions
                    if feature_def['type'] == 'categorical' and 'value_descriptions' in feature_def:
                        try:
                            value_int = int(value)
                            value_desc = feature_def['value_descriptions'].get(value_int, str(value))
                            desc = f"{description}: {value_desc}"
                        except (ValueError, TypeError):
                            desc = f"{description}: {value}"
                    else:
                        desc = f"{description}: {value}"
                    
                    factors.append({
                        'feature': feature,
                        'value': value,
                        'description': desc,
                        'importance': importance
                    })
        
        # If no factors found, add some default ones
        if not factors:
            for feature, value in input_data.iloc[0].items():
                factors.append({
                    'feature': feature,
                    'value': value,
                    'description': f"{feature}: {value}",
                    'importance': 0.1  # Default importance
                })
                
                if len(factors) >= top_n:
                    break
        
        # Sort by importance and take top_n
        factors.sort(key=lambda x: x['importance'], reverse=True)
        return factors[:top_n] 