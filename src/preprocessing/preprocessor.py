"""
Preprocessing module for heart disease prediction.
This module handles data preprocessing tasks such as feature scaling,
encoding categorical variables, and handling missing values.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.data.feature_definitions import FEATURES, DERIVED_FEATURES

class HeartDiseasePreprocessor:
    """
    Class for preprocessing heart disease data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with feature definitions."""
        self.features = FEATURES
        self.derived_features = DERIVED_FEATURES
        self.numeric_features = []
        self.categorical_features = []
        self._identify_feature_types()
        self.preprocessor = None
        
    def _identify_feature_types(self):
        """Identify numeric and categorical features from feature definitions."""
        for feature, props in self.features.items():
            if props['type'] == 'numeric':
                self.numeric_features.append(feature)
            elif props['type'] == 'categorical':
                self.categorical_features.append(feature)
    
    def _calculate_derived_features(self, data):
        """Calculate derived features based on their definitions."""
        for feature, props in self.derived_features.items():
            # Check if all source features are present
            if all(src in data.columns for src in props['source_features']):
                # Create a local namespace with the required features
                namespace = {src: data[src] for src in props['source_features']}
                # Evaluate the calculation expression in the namespace
                data[feature] = eval(props['calculation'], {"__builtins__": {}}, namespace)
        return data
    
    def fit(self, X):
        """
        Fit the preprocessor to the training data.
        
        Args:
            X (pd.DataFrame): Training data
            
        Returns:
            self: The fitted preprocessor
        """
        # Calculate derived features
        X = self._calculate_derived_features(X)
        
        # Add derived numeric features to numeric_features list
        for feature, props in self.derived_features.items():
            if props['type'] == 'numeric':
                self.numeric_features.append(feature)
        
        # Create preprocessing pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine the transformers using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )
        
        # Fit the preprocessor to the data
        self.preprocessor.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted preprocessor.
        
        Args:
            X (pd.DataFrame): Data to transform
            
        Returns:
            np.ndarray: Transformed data
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        # Ensure all required columns are present
        X_copy = X.copy()
        
        # Add missing columns with default values
        for feature in self.numeric_features + self.categorical_features:
            if feature not in X_copy.columns:
                if feature in self.features:
                    # For numeric features, use median of range
                    if self.features[feature]['type'] == 'numeric' and 'range' in self.features[feature]:
                        default_value = sum(self.features[feature]['range']) / 2
                    # For categorical features, use first value
                    elif self.features[feature]['type'] == 'categorical' and 'values' in self.features[feature]:
                        default_value = self.features[feature]['values'][0]
                    else:
                        default_value = 0
                    X_copy[feature] = default_value
                    print(f"Warning: Added missing column '{feature}' with default value {default_value}")
        
        # Calculate derived features
        X_copy = self._calculate_derived_features(X_copy)
        
        # Transform the data
        X_transformed = self.preprocessor.transform(X_copy)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit the preprocessor to the data and transform it.
        
        Args:
            X (pd.DataFrame): Data to fit and transform
            
        Returns:
            np.ndarray: Transformed data
        """
        return self.fit(X).transform(X)
    
    def validate_input(self, data):
        """
        Validate input data against feature definitions.
        
        Args:
            data (dict): Input data to validate
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        error_messages = []
        
        # Check for required features
        for feature, props in self.features.items():
            if props.get('required', False) and feature not in data:
                error_messages.append(f"Missing required feature: {feature}")
        
        # Validate feature values
        for feature, value in data.items():
            if feature in self.features:
                props = self.features[feature]
                
                # Validate numeric features
                if props['type'] == 'numeric':
                    try:
                        value = float(value)
                        if 'range' in props and (value < props['range'][0] or value > props['range'][1]):
                            error_messages.append(
                                f"Value for {feature} ({value}) is outside valid range "
                                f"[{props['range'][0]}, {props['range'][1]}]"
                            )
                    except (ValueError, TypeError):
                        error_messages.append(f"Invalid numeric value for {feature}: {value}")
                
                # Validate categorical features
                elif props['type'] == 'categorical':
                    try:
                        value = int(value)
                        if 'values' in props and value not in props['values']:
                            error_messages.append(
                                f"Invalid categorical value for {feature}: {value}. "
                                f"Valid values are {props['values']}"
                            )
                    except (ValueError, TypeError):
                        error_messages.append(f"Invalid categorical value for {feature}: {value}")
        
        return len(error_messages) == 0, error_messages 