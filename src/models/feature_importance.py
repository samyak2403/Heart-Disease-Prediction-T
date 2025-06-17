"""
Feature importance visualization for heart disease prediction.
This module provides functions for visualizing feature importance from the trained model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_importance(model, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained HeartDiseaseModel instance
        top_n (int): Number of top features to show
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Get feature importance
    feature_importance = model.get_feature_importance()
    
    # Convert to DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    
    # Add labels and title
    ax.set_title(f'Top {top_n} Features by Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_feature_impact(model, input_data, figsize=(10, 6)):
    """
    Plot the impact of features on a specific prediction.
    
    Args:
        model: Trained HeartDiseaseModel instance
        input_data (dict or pd.DataFrame): Input data for prediction
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Get explanation
    explanation = model.explain_prediction(input_data)
    
    # Check if contributing factors are available
    if 'contributing_factors' not in explanation:
        raise ValueError("Model explanation does not include contributing factors.")
    
    # Extract contributing factors
    factors = explanation['contributing_factors']
    
    # Create DataFrame for plotting
    impact_df = pd.DataFrame({
        'Feature': [f['description'] for f in factors],
        'Importance': [f['importance'] for f in factors]
    })
    
    # Sort by importance
    impact_df = impact_df.sort_values('Importance', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = sns.barplot(x='Importance', y='Feature', data=impact_df, ax=ax)
    
    # Color bars based on prediction
    if explanation['prediction'] == 1:
        # High risk - use red color gradient
        colors = sns.color_palette("Reds_r", len(impact_df))
    else:
        # Low risk - use green color gradient
        colors = sns.color_palette("Greens_r", len(impact_df))
    
    # Apply colors to bars
    for i, bar in enumerate(bars.patches):
        bar.set_color(colors[i])
    
    # Add title and labels
    risk_text = "High" if explanation['prediction'] == 1 else "Low"
    ax.set_title(f'Feature Impact on {risk_text} Risk Prediction ({explanation["probability"]:.1%} probability)')
    ax.set_xlabel('Impact')
    ax.set_ylabel('Feature')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def get_top_features(model, n=10):
    """
    Get the top n most important features from the model.
    
    Args:
        model: Trained HeartDiseaseModel instance
        n (int): Number of top features to return
        
    Returns:
        list: List of top feature names
    """
    feature_importance = model.get_feature_importance()
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    return [feature for feature, importance in sorted_features[:n]]


def get_feature_recommendations(model, threshold=0.02):
    """
    Get recommendations for which features to include in a UI based on importance.
    
    Args:
        model: Trained HeartDiseaseModel instance
        threshold (float): Minimum importance threshold for inclusion
        
    Returns:
        dict: Dictionary with feature categories and recommended features
    """
    # Get feature importance
    feature_importance = model.get_feature_importance()
    
    # Filter features by importance threshold
    important_features = {k: v for k, v in feature_importance.items() if v >= threshold}
    
    # Get feature definitions
    features = model.preprocessor.features
    
    # Categorize features
    demographic_features = []
    medical_history_features = []
    ecg_features = []
    exercise_features = []
    diagnostic_features = []
    
    # Define feature categories
    demographic_keys = ['age', 'sex', 'weight', 'height']
    medical_history_keys = ['cp', 'trestbps', 'chol', 'fbs', 'family_history']
    ecg_keys = ['restecg', 'thalach', 'exang']
    exercise_keys = ['oldpeak', 'slope']
    diagnostic_keys = ['ca', 'thal']
    
    # Categorize important features
    for feature in important_features:
        if feature in demographic_keys:
            demographic_features.append(feature)
        elif feature in medical_history_keys:
            medical_history_features.append(feature)
        elif feature in ecg_keys:
            ecg_features.append(feature)
        elif feature in exercise_keys:
            exercise_features.append(feature)
        elif feature in diagnostic_keys:
            diagnostic_features.append(feature)
    
    # Create recommendations
    recommendations = {
        'demographic': demographic_features,
        'medical_history': medical_history_features,
        'ecg': ecg_features,
        'exercise': exercise_features,
        'diagnostic': diagnostic_features,
        'all_important': list(important_features.keys()),
        'importances': important_features
    }
    
    return recommendations


def analyze_feature_importance(model, top_n=10, threshold=0.02):
    """
    Analyze feature importance and print recommendations for UI design.
    
    Args:
        model: Trained HeartDiseaseModel instance
        top_n (int): Number of top features to show
        threshold (float): Minimum importance threshold for inclusion
        
    Returns:
        dict: Feature recommendations
    """
    # Get feature importance
    feature_importance = model.get_feature_importance()
    
    # Convert to DataFrame for easier analysis
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    # Get top features
    top_features = importance_df.head(top_n)
    
    # Print analysis
    print(f"\n=== Feature Importance Analysis ===\n")
    print(f"Top {top_n} most important features:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Get recommendations
    recommendations = get_feature_recommendations(model, threshold)
    
    # Print recommendations by category
    print("\nRecommended features for UI by category:")
    
    # Print demographic features
    if recommendations['demographic']:
        print("\nDemographic features:")
        for feature in recommendations['demographic']:
            importance = feature_importance[feature]
            print(f"- {feature}: {importance:.4f}")
    
    # Print medical history features
    if recommendations['medical_history']:
        print("\nMedical history features:")
        for feature in recommendations['medical_history']:
            importance = feature_importance[feature]
            print(f"- {feature}: {importance:.4f}")
    
    # Print ECG features
    if recommendations['ecg']:
        print("\nECG features:")
        for feature in recommendations['ecg']:
            importance = feature_importance[feature]
            print(f"- {feature}: {importance:.4f}")
    
    # Print exercise features
    if recommendations['exercise']:
        print("\nExercise test features:")
        for feature in recommendations['exercise']:
            importance = feature_importance[feature]
            print(f"- {feature}: {importance:.4f}")
    
    # Print diagnostic features
    if recommendations['diagnostic']:
        print("\nAdditional diagnostic features:")
        for feature in recommendations['diagnostic']:
            importance = feature_importance[feature]
            print(f"- {feature}: {importance:.4f}")
    
    return recommendations 