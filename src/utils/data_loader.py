"""
Data loader utility for heart disease prediction.
This module provides functions for loading and processing heart disease datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_sample_data():
    """
    Load a sample heart disease dataset.
    This is a placeholder for demonstration purposes.
    In a real application, this would load data from a file or database.
    
    Returns:
        pd.DataFrame: Sample heart disease dataset
    """
    # Create a sample dataset with synthetic data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        # Demographic Features
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'weight': np.random.normal(75, 15, n_samples),
        'height': np.random.normal(170, 10, n_samples),
        
        # Medical History
        'cp': np.random.randint(0, 4, n_samples),  # chest pain type
        'trestbps': np.random.normal(130, 20, n_samples),  # resting blood pressure
        'chol': np.random.normal(220, 40, n_samples),  # cholesterol
        'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
        'family_history': np.random.randint(0, 2, n_samples),
        
        # ECG Features
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.normal(150, 20, n_samples),  # max heart rate
        'exang': np.random.randint(0, 2, n_samples),  # exercise induced angina
        
        # Exercise Test Features
        'oldpeak': np.random.normal(1, 1, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        
        # Additional Diagnostic Features
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 3, n_samples),
        
        # Target variable
        'target': np.random.randint(0, 2, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure proper data types
    df['age'] = df['age'].astype(int)
    df['sex'] = df['sex'].astype(int)
    df['weight'] = df['weight'].round(1)
    df['height'] = df['height'].round(1)
    df['cp'] = df['cp'].astype(int)
    df['trestbps'] = df['trestbps'].round().astype(int)
    df['chol'] = df['chol'].round().astype(int)
    df['fbs'] = df['fbs'].astype(int)
    df['family_history'] = df['family_history'].astype(int)
    df['restecg'] = df['restecg'].astype(int)
    df['thalach'] = df['thalach'].round().astype(int)
    df['exang'] = df['exang'].astype(int)
    df['oldpeak'] = df['oldpeak'].round(1)
    df['slope'] = df['slope'].astype(int)
    df['ca'] = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)
    df['target'] = df['target'].astype(int)
    
    return df


def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input data
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of training data for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    # Adjust validation size to account for the test split
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=adjusted_val_size, 
        random_state=random_state,
        stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_example_input():
    """
    Generate an example input for prediction.
    
    Returns:
        dict: Example input data
    """
    example_input = {
        'age': 52,
        'sex': 1,  # Male
        'weight': 80,
        'height': 175,
        'cp': 1,  # Atypical angina
        'trestbps': 125,
        'chol': 212,
        'fbs': 0,  # No
        'family_history': 1,  # Yes
        'restecg': 0,  # Normal
        'thalach': 168,
        'exang': 0,  # No
        'oldpeak': 1.0,
        'slope': 0,  # Upsloping
        'ca': 2,
        'thal': 0  # Normal
    }
    
    return example_input 