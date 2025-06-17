"""
Gradio interface for heart disease prediction.
This module provides a web interface for users to interact with the heart disease prediction model.
"""

import os
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.model import HeartDiseaseModel
from src.models.feature_importance import plot_feature_impact, get_feature_recommendations
from src.data.feature_definitions import FEATURES
from src.utils.data_loader import generate_example_input


def load_or_train_model(model_path='models/heart_disease_model.joblib'):
    """
    Load an existing model or train a new one.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        HeartDiseaseModel: The loaded or trained model
    """
    try:
        model = HeartDiseaseModel.load(model_path)
        print(f"Model loaded from {model_path}")
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
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"New model trained and saved to {model_path}")
    
    return model


def get_important_features(model, threshold=0.02):
    """
    Get important features based on feature importance.
    
    Args:
        model (HeartDiseaseModel): Trained model
        threshold (float): Importance threshold
        
    Returns:
        list: List of important feature names
    """
    recommendations = get_feature_recommendations(model, threshold)
    return recommendations['all_important']


def create_ui(model_path='models/heart_disease_model.joblib'):
    """
    Create a Gradio interface for heart disease prediction.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        gr.Interface: Gradio interface
    """
    # Load or train the model
    model = load_or_train_model(model_path)
    
    # Get important features
    important_features = get_important_features(model)
    
    # Create input components based on important features
    input_components = []
    feature_names = []
    
    # Add all required features and important features
    for feature, props in FEATURES.items():
        if props.get('required', True) or feature in important_features:
            feature_names.append(feature)
            
            # Create appropriate input component based on feature type
            if props['type'] == 'numeric':
                if 'range' in props:
                    input_comp = gr.Number(
                        label=props['description'],
                        minimum=props['range'][0],
                        maximum=props['range'][1],
                        value=(props['range'][0] + props['range'][1]) / 2  # Default to middle of range
                    )
                else:
                    input_comp = gr.Number(label=props['description'])
            elif props['type'] == 'categorical':
                if 'value_descriptions' in props:
                    choices = [f"{value}: {desc}" for value, desc in props['value_descriptions'].items()]
                    input_comp = gr.Dropdown(
                        label=props['description'],
                        choices=choices,
                        value=choices[0] if choices else None
                    )
                else:
                    input_comp = gr.Dropdown(
                        label=props['description'],
                        choices=[str(v) for v in props['values']],
                        value=str(props['values'][0]) if props['values'] else None
                    )
            else:
                input_comp = gr.Textbox(label=props['description'])
                
            input_components.append(input_comp)
    
    # Define prediction function
    def predict(*args):
        # Create input dictionary
        input_data = {}
        for i, feature in enumerate(feature_names):
            value = args[i]
            
            # Extract value from dropdown if needed
            if isinstance(value, str) and ':' in value:
                value = int(value.split(':', 1)[0].strip())
                
            input_data[feature] = value
        
        # Make prediction
        explanation = model.explain_prediction(input_data)
        
        # Format result
        if explanation['prediction'] == 1:
            result = f"High risk of heart disease ({explanation['probability']:.1%} probability)"
        else:
            result = f"Low risk of heart disease ({1-explanation['probability']:.1%} probability)"
        
        risk_level = f"Risk Level: {explanation['risk_level']}"
        
        # Format contributing factors
        factors = []
        if 'contributing_factors' in explanation:
            for factor in explanation['contributing_factors']:
                factors.append(f"- {factor['description']}")
        
        contributing_factors = "\n".join(factors)
        
        # Generate plot
        try:
            fig = plot_feature_impact(model, input_data)
            plt.close(fig)  # Close the figure to avoid memory leaks
        except Exception as e:
            fig = None
            print(f"Could not generate plot: {e}")
        
        return result, risk_level, contributing_factors, fig
    
    # Create interface
    iface = gr.Interface(
        fn=predict,
        inputs=input_components,
        outputs=[
            gr.Textbox(label="Prediction Result"),
            gr.Textbox(label="Risk Level"),
            gr.Textbox(label="Contributing Factors"),
            gr.Plot(label="Feature Impact")
        ],
        title="Heart Disease Risk Prediction",
        description="Enter patient information to predict heart disease risk.",
        examples=[[generate_example_input()[feature] for feature in feature_names]],
        theme="default"
    )
    
    return iface


def main():
    """
    Run the Gradio interface.
    """
    iface = create_ui()
    iface.launch(share=True)


if __name__ == "__main__":
    main() 