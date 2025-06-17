"""
Console interface for heart disease prediction.
This module provides a command-line interface for users to interact with the heart disease prediction model.
"""

import pandas as pd
from src.data.feature_definitions import FEATURES
from src.models.model import HeartDiseaseModel

class ConsoleInterface:
    """
    Console interface for heart disease prediction.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the console interface.
        
        Args:
            model_path (str, optional): Path to a saved model
        """
        if model_path:
            self.model = HeartDiseaseModel.load(model_path)
        else:
            self.model = None
        
    def collect_input(self):
        """
        Collect user input for all required features.
        
        Returns:
            dict: Collected feature values
        """
        print("\n=== Heart Disease Risk Assessment ===\n")
        print("Please enter the following information:\n")
        
        user_input = {}
        
        # Collect input for each required feature
        for feature, props in FEATURES.items():
            if props.get('required', False):
                valid_input = False
                while not valid_input:
                    # Display prompt with description
                    prompt = f"{props['description']}: "
                    
                    # For categorical features, show available options
                    if props['type'] == 'categorical' and 'value_descriptions' in props:
                        prompt += "\n"
                        for value, desc in props['value_descriptions'].items():
                            prompt += f"  {value}: {desc}\n"
                        prompt += "Enter value: "
                    
                    # Get user input
                    value = input(prompt)
                    
                    # Validate input
                    try:
                        if props['type'] == 'numeric':
                            value = float(value)
                            if 'range' in props and (value < props['range'][0] or value > props['range'][1]):
                                print(f"Value must be between {props['range'][0]} and {props['range'][1]}.")
                                continue
                        elif props['type'] == 'categorical':
                            value = int(value)
                            if 'values' in props and value not in props['values']:
                                print(f"Value must be one of: {props['values']}")
                                continue
                        
                        user_input[feature] = value
                        valid_input = True
                    except ValueError:
                        print("Invalid input. Please try again.")
        
        return user_input
    
    def predict_and_explain(self, user_input):
        """
        Make a prediction and explain it to the user.
        
        Args:
            user_input (dict): User input data
        """
        if self.model is None:
            print("Error: Model not loaded.")
            return
        
        # Make prediction
        explanation = self.model.explain_prediction(user_input)
        
        # Display results
        print("\n=== Heart Disease Risk Assessment Results ===\n")
        
        if explanation['prediction'] == 1:
            print(f"Result: High risk of heart disease ({explanation['probability']:.1%} probability)")
        else:
            print(f"Result: Low risk of heart disease ({1-explanation['probability']:.1%} probability)")
        
        print(f"Risk Level: {explanation['risk_level']}")
        
        if 'contributing_factors' in explanation:
            print("\nKey Contributing Factors:")
            for factor in explanation['contributing_factors']:
                print(f"- {factor['description']}")
        
        print("\nNote: This is a screening tool only and not a medical diagnosis.")
        print("Please consult with a healthcare professional for proper medical advice.")
    
    def run(self):
        """
        Run the console interface.
        """
        if self.model is None:
            print("Error: Model not loaded.")
            return
        
        while True:
            # Collect user input
            user_input = self.collect_input()
            
            # Make prediction and explain
            self.predict_and_explain(user_input)
            
            # Ask if user wants to continue
            if input("\nWould you like to make another prediction? (y/n): ").lower() != 'y':
                break
        
        print("\nThank you for using the Heart Disease Risk Assessment tool.")


def main(model_path='models/heart_disease_model.joblib'):
    """
    Main function to run the console interface.
    
    Args:
        model_path (str): Path to the saved model
    """
    interface = ConsoleInterface(model_path)
    interface.run()


if __name__ == '__main__':
    main() 