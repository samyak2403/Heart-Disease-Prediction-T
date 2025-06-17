"""
Main script for heart disease prediction system.
This script demonstrates the heart disease prediction system by training a model
on sample data and making predictions.
"""

import os
import pandas as pd
import joblib

from src.utils.data_loader import load_sample_data, split_data, generate_example_input
from src.models.model import HeartDiseaseModel
from src.ui.console_interface import ConsoleInterface


def train_model(save_path='models/heart_disease_model.joblib'):
    """
    Train a heart disease prediction model on sample data.
    
    Args:
        save_path (str): Path to save the trained model
        
    Returns:
        HeartDiseaseModel: Trained model
    """
    print("Loading sample data...")
    df = load_sample_data()
    
    print(f"Sample data shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create and train model
    print("Training model...")
    model = HeartDiseaseModel(model_type='random_forest')
    model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\nValidation metrics:")
    for metric, value in val_metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            print(f"  {metric}: {value:.4f}")
    
    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            print(f"  {metric}: {value:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model


def test_prediction(model=None, model_path='models/heart_disease_model.joblib'):
    """
    Test the prediction functionality with an example input.
    
    Args:
        model (HeartDiseaseModel, optional): Trained model
        model_path (str): Path to load the model if not provided
    """
    if model is None:
        print(f"Loading model from {model_path}...")
        model = HeartDiseaseModel.load(model_path)
    
    # Generate example input
    example_input = generate_example_input()
    
    print("\nExample input:")
    for feature, value in example_input.items():
        print(f"  {feature}: {value}")
    
    # Make prediction
    print("\nMaking prediction...")
    explanation = model.explain_prediction(example_input)
    
    # Display results
    print("\nPrediction results:")
    if explanation['prediction'] == 1:
        print(f"  Result: High risk of heart disease ({explanation['probability']:.1%} probability)")
    else:
        print(f"  Result: Low risk of heart disease ({1-explanation['probability']:.1%} probability)")
    
    print(f"  Risk Level: {explanation['risk_level']}")
    
    if 'contributing_factors' in explanation:
        print("\nKey Contributing Factors:")
        for factor in explanation['contributing_factors']:
            print(f"  - {factor['description']}")


def run_console_interface(model_path='models/heart_disease_model.joblib'):
    """
    Run the console interface for user interaction.
    
    Args:
        model_path (str): Path to the trained model
    """
    interface = ConsoleInterface(model_path)
    interface.run()


def main():
    """
    Main function to run the heart disease prediction system.
    """
    print("=== Heart Disease Prediction System ===\n")
    
    # Check if model exists
    model_path = 'models/heart_disease_model.joblib'
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        use_existing = input("Use existing model? (y/n): ").lower() == 'y'
        
        if use_existing:
            # Test prediction with existing model
            test_prediction(model_path=model_path)
        else:
            # Train new model
            model = train_model(model_path)
            test_prediction(model=model)
    else:
        print("No existing model found. Training new model...")
        model = train_model(model_path)
        test_prediction(model=model)
    
    # Ask if user wants to use console interface
    use_console = input("\nWould you like to use the interactive console interface? (y/n): ").lower() == 'y'
    if use_console:
        run_console_interface(model_path)


if __name__ == "__main__":
    main() 