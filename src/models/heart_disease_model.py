"""
Heart disease prediction model.
This module contains the model for predicting heart disease.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

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
        self.feature_importances_ = None
        
    def _create_model(self):
        """
        Create the model based on model_type.
        
        Returns:
            object: The created model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X, y, optimize_hyperparams=False):
        """
        Train the model on the given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            optimize_hyperparams (bool): Whether to optimize hyperparameters
            
        Returns:
            self: The trained model
        """
        # Preprocess the data
        X_preprocessed = self.preprocessor.fit_transform(X)
        
        if optimize_hyperparams:
            # Define hyperparameter grid
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Create a base model
                base_model = self._create_model()
                
                # Create grid search
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                # Fit grid search
                grid_search.fit(X_preprocessed, y)
                
                # Get best model
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {self.model_type}")
        else:
            # Create and train the model
            self.model = self._create_model()
            self.model.fit(X_preprocessed, y)
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
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
        
        # Preprocess the data
        X_preprocessed = self.preprocessor.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_preprocessed)
        y_prob = self.model.predict_proba(X_preprocessed)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred)
        }
        
        return metrics
    
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
        
        # Get feature names
        feature_names = []
        
        # Add numeric feature names
        for name in self.preprocessor.numeric_features:
            feature_names.append(name)
        
        # Add categorical feature names with one-hot encoding
        for i, name in enumerate(self.preprocessor.categorical_features):
            if hasattr(self.preprocessor.preprocessor.transformers_[1][1][-1], 'get_feature_names_out'):
                cat_features = self.preprocessor.preprocessor.transformers_[1][1][-1].get_feature_names_out([name])
                feature_names.extend(cat_features)
            else:
                # Fallback for older scikit-learn versions
                values = self.preprocessor.features[name].get('values', [])
                for val in values:
                    feature_names.append(f"{name}_{val}")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a dictionary of feature importances
        feature_importance = {}
        for i, name in enumerate(feature_names):
            if i < len(importances):
                feature_importance[name] = importances[i]
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
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
            'model_type': self.model_type,
            'feature_importances': self.feature_importances_
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
        instance.feature_importances_ = model_data['feature_importances']
        
        return instance
    
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
        
        # Get feature importances
        feature_importance = self.get_feature_importance()
        
        # Create explanation
        explanation = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': self._get_risk_level(probability),
            'contributing_factors': self._get_contributing_factors(input_data, feature_importance)
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
    
    def _get_contributing_factors(self, input_data, feature_importance, top_n=5):
        """
        Get top contributing factors for the prediction.
        
        Args:
            input_data (pd.DataFrame): Input data
            feature_importance (dict): Feature importance scores
            top_n (int): Number of top factors to return
            
        Returns:
            list: Top contributing factors
        """
        factors = []
        
        # Get top features by importance
        top_features = list(feature_importance.keys())[:top_n*2]  # Get more than needed to filter
        
        # For each feature, check if it's a risk factor
        for feature in top_features:
            # Handle one-hot encoded features
            if '_' in feature:
                base_feature, value = feature.rsplit('_', 1)
                try:
                    value = int(value)
                    if base_feature in input_data.columns and input_data[base_feature].iloc[0] == value:
                        # Get the description from feature definitions
                        if base_feature in self.preprocessor.features:
                            feature_def = self.preprocessor.features[base_feature]
                            description = feature_def.get('description', base_feature)
                            
                            # Get value description if available
                            value_desc = feature_def.get('value_descriptions', {}).get(value, str(value))
                            
                            factors.append({
                                'feature': base_feature,
                                'value': value,
                                'description': f"{description}: {value_desc}",
                                'importance': feature_importance[feature]
                            })
                except ValueError:
                    pass
            else:
                # Handle numeric features
                if feature in input_data.columns:
                    value = input_data[feature].iloc[0]
                    
                    # Get the description from feature definitions
                    if feature in self.preprocessor.features:
                        feature_def = self.preprocessor.features[feature]
                        description = feature_def.get('description', feature)
                        
                        factors.append({
                            'feature': feature,
                            'value': float(value),
                            'description': f"{description}: {value}",
                            'importance': feature_importance[feature]
                        })
                    elif feature in self.preprocessor.derived_features:
                        feature_def = self.preprocessor.derived_features[feature]
                        description = feature_def.get('description', feature)
                        
                        factors.append({
                            'feature': feature,
                            'value': float(value),
                            'description': f"{description}: {value}",
                            'importance': feature_importance[feature]
                        })
        
        # Sort by importance and take top_n
        factors.sort(key=lambda x: x['importance'], reverse=True)
        return factors[:top_n] 