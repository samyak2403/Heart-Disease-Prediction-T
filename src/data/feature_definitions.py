"""
Feature definitions for heart disease prediction.
This module defines the features used in the heart disease prediction model,
including their descriptions, types, and valid ranges.
"""

# Dictionary containing feature definitions with descriptions, types, and valid ranges
FEATURES = {
    # Demographic Features
    "age": {
        "description": "Age in years",
        "type": "numeric",
        "range": [20, 100],
        "required": True
    },
    "sex": {
        "description": "Gender (1 = male, 0 = female)",
        "type": "categorical",
        "values": [0, 1],
        "required": True
    },
    "weight": {
        "description": "Weight in kg",
        "type": "numeric",
        "range": [30, 250],
        "required": True
    },
    "height": {
        "description": "Height in cm",
        "type": "numeric",
        "range": [120, 220],
        "required": True
    },
    
    # Medical History
    "cp": {
        "description": "Chest pain type",
        "type": "categorical",
        "values": [0, 1, 2, 3],
        "value_descriptions": {
            0: "Typical angina",
            1: "Atypical angina",
            2: "Non-anginal pain",
            3: "Asymptomatic"
        },
        "required": True
    },
    "trestbps": {
        "description": "Resting blood pressure (mm Hg)",
        "type": "numeric",
        "range": [80, 200],
        "required": True
    },
    "chol": {
        "description": "Serum cholesterol (mg/dl)",
        "type": "numeric",
        "range": [100, 600],
        "required": True
    },
    "fbs": {
        "description": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "type": "categorical",
        "values": [0, 1],
        "required": True
    },
    "family_history": {
        "description": "Family history of heart disease (1 = yes, 0 = no)",
        "type": "categorical",
        "values": [0, 1],
        "required": False
    },
    
    # ECG Features
    "restecg": {
        "description": "Resting electrocardiographic results",
        "type": "categorical",
        "values": [0, 1, 2],
        "value_descriptions": {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy"
        },
        "required": True
    },
    "thalach": {
        "description": "Maximum heart rate achieved",
        "type": "numeric",
        "range": [60, 220],
        "required": True
    },
    "exang": {
        "description": "Exercise induced angina (1 = yes, 0 = no)",
        "type": "categorical",
        "values": [0, 1],
        "required": True
    },
    
    # Exercise Test Features
    "oldpeak": {
        "description": "ST depression induced by exercise relative to rest",
        "type": "numeric",
        "range": [0, 10],
        "required": True
    },
    "slope": {
        "description": "Slope of the peak exercise ST segment",
        "type": "categorical",
        "values": [0, 1, 2],
        "value_descriptions": {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        },
        "required": True
    },
    
    # Additional Diagnostic Features
    "ca": {
        "description": "Number of major vessels colored by fluoroscopy (0-3)",
        "type": "categorical",
        "values": [0, 1, 2, 3],
        "required": True
    },
    "thal": {
        "description": "Thalassemia",
        "type": "categorical",
        "values": [0, 1, 2],
        "value_descriptions": {
            0: "Normal",
            1: "Fixed defect",
            2: "Reversible defect"
        },
        "required": True
    }
}

# Derived features that are calculated from other features
DERIVED_FEATURES = {
    "bmi": {
        "description": "Body Mass Index (kg/m²)",
        "type": "numeric",
        "source_features": ["weight", "height"],
        "calculation": "weight / ((height/100) ** 2)"
    }
}

# Target variable
TARGET = {
    "target": {
        "description": "Presence of heart disease (1 = yes, 0 = no)",
        "type": "binary",
        "values": [0, 1]
    }
} 