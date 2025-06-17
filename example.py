"""
Example script demonstrating the heart disease prediction system.
This script shows how to use the system to make a prediction for a sample patient.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.models.model import HeartDiseaseModel
from src.utils.data_loader import generate_example_input
from src.models.feature_importance import plot_feature_impact

def main():
    """
    Run an example prediction with the heart disease model.
    """
    print("=== Heart Disease Prediction Example ===\n")
    
    # Sample patient data
    patient_data = {
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
    
    # Print patient information
    print("Patient Information:")
    print(f"  Age: {patient_data['age']} years")
    print(f"  Sex: {'Male' if patient_data['sex'] == 1 else 'Female'}")
    print(f"  Weight: {patient_data['weight']} kg")
    print(f"  Height: {patient_data['height']} cm")
    print(f"  Blood Pressure: {patient_data['trestbps']} mm Hg")
    print(f"  Cholesterol: {patient_data['chol']} mg/dl")
    
    try:
        # Load or train model
        try:
            print("\nLoading model...")
            model = HeartDiseaseModel.load('models/heart_disease_model.joblib')
            print("Model loaded successfully.")
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load model: {e}")
            print("Training a new model...")
            
            # Import here to avoid circular imports
            from src.utils.data_loader import load_sample_data, split_data
            
            # Load sample data
            df = load_sample_data()
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
            
            # Create and train model
            model = HeartDiseaseModel(model_type='random_forest')
            model.train(X_train, y_train)
            
            # Save model
            import os
            os.makedirs('models', exist_ok=True)
            model.save('models/heart_disease_model.joblib')
            print("New model trained and saved.")
        
        # Make prediction
        print("\nMaking prediction...")
        explanation = model.explain_prediction(patient_data)
        
        # Display results
        print("\nPrediction Results:")
        if explanation['prediction'] == 1:
            print(f"  Result: High risk of heart disease ({explanation['probability']:.1%} probability)")
        else:
            print(f"  Result: Low risk of heart disease ({1-explanation['probability']:.1%} probability)")
        
        print(f"  Risk Level: {explanation['risk_level']}")
        
        if 'contributing_factors' in explanation:
            print("\nKey Contributing Factors:")
            for factor in explanation['contributing_factors']:
                print(f"  - {factor['description']}")
        
        # Plot feature impact if matplotlib is available
        try:
            print("\nGenerating feature impact visualization...")
            fig = plot_feature_impact(model, patient_data)
            plt.show()
        except Exception as e:
            print(f"Could not generate visualization: {e}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("\nNote: This is a demonstration only and not for actual medical diagnosis.")

if __name__ == "__main__":
    main() 