"""
Standalone script for analyzing feature importance in the heart disease prediction model.
"""

import os
import sys
import matplotlib.pyplot as plt
from src.models.model import HeartDiseaseModel
from src.models.feature_importance import analyze_feature_importance, plot_feature_importance
from src.utils.data_loader import load_sample_data, split_data

def main():
    """Main function to analyze feature importance."""
    print("=== Heart Disease Feature Importance Analysis ===\n")
    
    # Check if model exists
    model_path = 'models/heart_disease_model.joblib'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = HeartDiseaseModel.load(model_path)
    else:
        print("No existing model found. Training new model...")
        
        # Load sample data
        print("Loading sample data...")
        df = load_sample_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        
        # Create and train model
        print("Training model...")
        model = HeartDiseaseModel(model_type='random_forest')
        model.train(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    recommendations = analyze_feature_importance(model, top_n=15, threshold=0.02)
    
    # Plot feature importance
    print("\nGenerating feature importance plot...")
    fig = plot_feature_importance(model, top_n=15)
    
    # Save plot
    plot_path = "feature_importance.png"
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Show plot
    plt.show()
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main() 