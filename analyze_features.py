"""
Feature importance analysis and Gradio UI launcher.
This script analyzes feature importance and launches the Gradio UI for heart disease prediction.
"""

import os
import matplotlib.pyplot as plt

from src.models.model import HeartDiseaseModel
from src.models.feature_importance import analyze_feature_importance, plot_feature_importance
from src.ui.gradio_interface import create_ui
from src.utils.data_loader import load_sample_data, split_data


def main():
    """
    Main function to analyze feature importance and launch the Gradio UI.
    """
    model_path = 'models/heart_disease_model.joblib'
    
    # Load or train model
    try:
        print("Loading model...")
        model = HeartDiseaseModel.load(model_path)
        print("Model loaded successfully.")
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load model: {e}")
        print("Training a new model...")
        
        # Load sample data
        df = load_sample_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        
        # Create and train model
        model = HeartDiseaseModel(model_type='random_forest')
        model.train(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print("New model trained and saved.")
    
    # Analyze feature importance
    recommendations = analyze_feature_importance(model, top_n=15, threshold=0.02)
    
    # Plot feature importance
    print("\nGenerating feature importance plot...")
    fig = plot_feature_importance(model, top_n=15)
    plt.show()
    
    # Launch Gradio UI
    print("\nLaunching Gradio UI...")
    iface = create_ui(model_path)
    iface.launch(share=True)


if __name__ == "__main__":
    main() 