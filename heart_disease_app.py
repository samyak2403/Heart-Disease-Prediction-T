"""
Heart Disease Prediction System - Executable Application
This script provides direct access to the Gradio web interface.
"""

import os
import sys
import webbrowser
import threading
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to path to ensure imports work in PyInstaller
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path adjustment
from src.models.model import HeartDiseaseModel
from src.utils.data_loader import load_sample_data, split_data
import gradio as gr

# Model path
MODEL_PATH = 'models/heart_disease_model.joblib'

def ensure_model_exists():
    """Ensure the model exists, train if it doesn't."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training a new model...")
        # Load sample data
        df = load_sample_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        
        # Create and train model
        model = HeartDiseaseModel(model_type='random_forest')
        model.train(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return model
    else:
        print(f"Loading model from {MODEL_PATH}")
        return HeartDiseaseModel.load(MODEL_PATH)

def calculate_bmi(height, weight):
    """Calculate BMI from height (cm) and weight (kg)."""
    # Convert height from cm to m
    height_m = height / 100
    # Calculate BMI
    bmi = weight / (height_m * height_m)
    return bmi

def get_bmi_category(bmi):
    """Get BMI category based on BMI value."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_feature_impacts(input_data, model):
    """Generate feature impact data with accurate medical reference values."""
    # Feature names with descriptions, educational information, and reference ranges
    feature_info = {
        'age': {
            'display': 'Age (years)',
            'info': 'Risk doubles every decade after 45',
            'normal_range': '20-45',
            'high_risk': '>65'
        },
        'sex': {
            'display': 'Gender',
            'info': 'Men have 2-3x higher risk than women before age 55',
            'normal_range': 'N/A',
            'high_risk': 'Male'
        },
        'cp': {
            'display': 'Chest Pain Type',
            'info': 'Typical angina strongly indicates coronary artery disease',
            'normal_range': 'None',
            'high_risk': 'Typical angina'
        },
        'trestbps': {
            'display': 'Resting Blood Pressure',
            'info': 'Each 20mmHg increase doubles risk',
            'normal_range': '90-120 mmHg',
            'high_risk': '>140 mmHg'
        },
        'chol': {
            'display': 'Serum Cholesterol',
            'info': '23% increased risk per 40mg/dl above 200',
            'normal_range': '<200 mg/dl',
            'high_risk': '>240 mg/dl'
        },
        'fbs': {
            'display': 'Fasting Blood Sugar',
            'info': 'Diabetes doubles heart disease risk',
            'normal_range': '<100 mg/dl',
            'high_risk': '>126 mg/dl'
        },
        'restecg': {
            'display': 'Resting ECG Results',
            'info': 'ST-T abnormalities indicate 5x higher risk',
            'normal_range': 'Normal',
            'high_risk': 'ST-T abnormality'
        },
        'thalach': {
            'display': 'Maximum Heart Rate',
            'info': 'Lower max heart rate indicates decreased function',
            'normal_range': '>150 bpm',
            'high_risk': '<120 bpm'
        },
        'exang': {
            'display': 'Exercise Induced Angina',
            'info': 'Presence indicates 3x higher risk',
            'normal_range': 'No',
            'high_risk': 'Yes'
        },
        'oldpeak': {
            'display': 'ST Depression by Exercise',
            'info': 'Values >2mm indicate severe ischemia',
            'normal_range': '<1mm',
            'high_risk': '>2mm'
        },
        'slope': {
            'display': 'Slope of Peak Exercise ST',
            'info': 'Downsloping indicates poor prognosis',
            'normal_range': 'Upsloping',
            'high_risk': 'Downsloping'
        },
        'ca': {
            'display': 'Number of Major Vessels',
            'info': 'Risk increases 2x per vessel affected',
            'normal_range': '0',
            'high_risk': '≥2'
        },
        'thal': {
            'display': 'Thalassemia',
            'info': 'Reversible defects indicate 3x higher risk',
            'normal_range': 'Normal',
            'high_risk': 'Reversible defect'
        }
    }
    
    # Get feature importances from the model's explanation
    explanation = model.explain_prediction(input_data)
    
    # Extract contributing factors
    impacts = []
    for factor in explanation['contributing_factors']:
        feature = factor['feature']
        if feature in feature_info:
            display_name = feature_info[feature]['display']
            info = feature_info[feature]['info']
            normal_range = feature_info[feature]['normal_range']
            high_risk = feature_info[feature]['high_risk']
        else:
            display_name = feature
            info = ""
            normal_range = "N/A"
            high_risk = "N/A"
            
        impacts.append({
            'name': display_name,
            'impact': factor['importance'],
            'info': info,
            'value': factor['value'],
            'normal_range': normal_range,
            'high_risk': high_risk
        })
    
    # Add BMI impact if not already included
    bmi_found = any('BMI' in item['name'] for item in impacts)
    if not bmi_found:
        bmi_value = calculate_bmi(input_data.get('height', 170), input_data.get('weight', 70))
        bmi_impact = 0.05  # Base impact value
        
        # Adjust impact based on BMI value
        if bmi_value >= 30:  # Obese
            bmi_impact = 0.15  # Higher impact for obesity
        elif bmi_value >= 25:  # Overweight
            bmi_impact = 0.1  # Medium impact for overweight
            
        impacts.append({
            'name': 'BMI',
            'impact': bmi_impact,
            'info': 'BMI >30 increases risk by 50%',
            'value': bmi_value,
            'normal_range': '18.5-25',
            'high_risk': '>30'
        })
    
    return impacts

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, height, weight):
    """Make a heart disease prediction based on input features."""
    # Calculate BMI
    bmi = calculate_bmi(height, weight)
    bmi_category = get_bmi_category(bmi)
    
    # Create input dictionary
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'height': height,
        'weight': weight
    }
    
    # Make prediction
    explanation = model.explain_prediction(input_data)
    
    # Format results
    if explanation['prediction'] == 1:
        result = f"High risk of heart disease ({explanation['probability']:.1%} probability)"
        risk_level = explanation['risk_level']
        risk_color = "#c0392b"  # Red for high risk
    else:
        result = f"Low risk of heart disease ({1-explanation['probability']:.1%} probability)"
        risk_level = explanation['risk_level']
        risk_color = "#27ae60"  # Green for low risk
    
    # Format contributing factors
    contributing_factors = ""
    if 'contributing_factors' in explanation:
        contributing_factors = "\n".join([f"• {factor['description']}" for factor in explanation['contributing_factors']])
    
    # Add BMI information to contributing factors
    bmi_info = f"\n\n• BMI: {bmi:.1f} ({bmi_category})"
    
    # Adjust risk based on BMI
    if bmi_category == "Obese" and "High risk" not in result:
        bmi_info += "\n• Obesity is a risk factor for heart disease, consider lifestyle changes."
    elif bmi_category == "Overweight" and "High risk" not in result:
        bmi_info += "\n• Being overweight may increase heart disease risk."
    elif bmi_category == "Underweight" and "High risk" not in result:
        bmi_info += "\n• Being underweight may indicate other health issues."
    
    contributing_factors += bmi_info
    
    # Generate feature impacts for visualization
    feature_impacts = get_feature_impacts(input_data, model)
    
    # Create feature impact visualization
    features = []
    impacts = []
    tooltips = []
    values = []
    normal_ranges = []
    high_risk_values = []
    
    for item in feature_impacts:
        feature_name = item['name']
        feature_value = item['value']
        
        # Format the feature name with value
        if isinstance(feature_value, (int, float)) and feature_value > 10:
            # For larger numbers, round to nearest integer
            feature_display = f"{feature_name}: {int(feature_value)}"
        elif isinstance(feature_value, float):
            # For smaller floating points, show 1 decimal
            feature_display = f"{feature_name}: {feature_value:.1f}"
        else:
            # For categorical values
            feature_display = f"{feature_name}: {feature_value}"
            
        features.append(feature_display)
        impacts.append(item['impact'])
        
        # Create detailed tooltip with normal and high risk ranges
        tooltip = f"{item['info']}\nNormal: {item['normal_range']}\nHigh Risk: {item['high_risk']}"
        tooltips.append(tooltip)
        
        # Store values for additional visualization
        values.append(feature_value)
        normal_ranges.append(item['normal_range'])
        high_risk_values.append(item['high_risk'])
    
    # Sort by absolute impact
    sorted_indices = np.argsort(np.abs(impacts))[-8:]  # Top 8 features
    sorted_features = [features[i] for i in sorted_indices]
    sorted_impacts = [impacts[i] for i in sorted_indices]
    sorted_tooltips = [tooltips[i] for i in sorted_indices]
    
    # Calculate total impact score for visualization
    total_positive_impact = sum(impact for impact in impacts if impact > 0)
    total_negative_impact = sum(impact for impact in impacts if impact < 0)
    total_impact = total_positive_impact + total_negative_impact
    
    # Create plot with larger font size
    plt.rcParams.update({'font.size': 26})  # Increased to 26
    fig = plt.figure(figsize=(16, 14))  # Increased height to accommodate total score
    
    # Create a grid for multiple plots
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
    
    # Main plot for feature impacts
    ax1 = plt.subplot(gs[0])
    
    # Use a more colorful style with better contrast
    plt.style.use('ggplot')
    
    # Create horizontal bar chart with custom colors based on whether the impact is positive or negative
    bar_colors = []
    for impact in sorted_impacts:
        if impact > 0:
            bar_colors.append('#c0392b')  # Red for risk-increasing factors
        else:
            bar_colors.append('#27ae60')  # Green for risk-decreasing factors
    
    bars = ax1.barh(sorted_features, sorted_impacts, color=bar_colors, height=0.8)
    
    ax1.set_xlabel("Impact on Prediction", fontsize=30, fontweight='bold')
    ax1.set_ylabel("Features", fontsize=30, fontweight='bold')
    ax1.set_title(f"Feature Impact on Heart Disease Risk", fontsize=36, fontweight='bold')
    
    # Add grid lines for better readability
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add a vertical line at x=0 to clearly show positive vs negative impacts
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=2)
    
    # Add value labels to bars with better positioning
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width >= 0 else width - 0.03
        ax1.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', fontsize=24, fontweight='bold',
                color='black' if width > 0 else 'white')
        
        # Add educational information as text annotations
        if i < len(sorted_tooltips):
            ax1.annotate(sorted_tooltips[i], 
                        xy=(0, bar.get_y() + bar.get_height()/2),
                        xytext=(-10, 0), 
                        textcoords="offset points",
                        ha="right", va="center",
                        fontsize=16, color='#444444',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.8))
    
    # Add legend for color meaning
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='#c0392b', edgecolor='none', label='Increases Risk'),
        plt.Rectangle((0, 0), 1, 1, fc='#27ae60', edgecolor='none', label='Decreases Risk')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=20)
    
    # Increase tick label size and add more padding
    ax1.tick_params(axis='both', which='major', labelsize=24, pad=10)
    
    # Add more space between subplot elements
    plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.15)
    
    # Create second subplot for total impact score
    ax2 = plt.subplot(gs[1])
    
    # Create a horizontal gauge chart for total risk score
    risk_score = explanation['probability'] * 100  # Convert to percentage
    gauge_colors = ['#27ae60', '#f39c12', '#c0392b']  # Green, Yellow, Red
    
    # Determine color based on risk score
    if risk_score < 30:
        gauge_color = gauge_colors[0]  # Green
    elif risk_score < 70:
        gauge_color = gauge_colors[1]  # Yellow
    else:
        gauge_color = gauge_colors[2]  # Red
    
    # Create gauge chart
    ax2.barh(['Total Risk Score'], [100], color='#e0e0e0', height=0.6)  # Background bar
    ax2.barh(['Total Risk Score'], [risk_score], color=gauge_color, height=0.6)  # Risk score bar
    
    # Add risk percentage text
    ax2.text(risk_score + 2, 0, f"{risk_score:.1f}%", 
            va='center', fontsize=28, fontweight='bold', color=gauge_color)
    
    # Add risk level text
    ax2.text(50, -0.5, f"Risk Level: {risk_level}", 
            ha='center', va='center', fontsize=24, fontweight='bold', color='#2c3e50')
    
    # Remove y-axis labels and ticks
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # Set x-axis limits and ticks
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=20)
    
    # Add reference markers for risk levels
    ax2.axvline(x=20, color='#27ae60', linestyle='--', alpha=0.7, linewidth=2)  # Low risk
    ax2.axvline(x=50, color='#f39c12', linestyle='--', alpha=0.7, linewidth=2)  # Moderate risk
    ax2.axvline(x=80, color='#c0392b', linestyle='--', alpha=0.7, linewidth=2)  # High risk
    
    # Add text labels for risk categories
    ax2.text(10, 0.8, "Low", ha='center', fontsize=18, color='#27ae60', fontweight='bold')
    ax2.text(35, 0.8, "Moderate", ha='center', fontsize=18, color='#f39c12', fontweight='bold')
    ax2.text(65, 0.8, "High", ha='center', fontsize=18, color='#c0392b', fontweight='bold')
    ax2.text(90, 0.8, "Very High", ha='center', fontsize=18, color='#7d3c98', fontweight='bold')
    
    # Add grid lines
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Set title
    ax2.set_title("Total Heart Disease Risk Score", fontsize=30, fontweight='bold')
    
    # Set background color for better visibility
    fig.patch.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    return result, risk_level, contributing_factors, fig

def create_interface():
    """Create the Gradio interface with larger text and better visibility."""
    # Custom CSS for larger text and better visibility
    custom_css = """
    .gradio-container {
        font-size: 22px !important;
        max-width: 1200px !important;
        margin: auto !important;
    }
    h1 {
        font-size: 46px !important;
        font-weight: bold !important;
        color: #e74c3c !important;
        text-align: center !important;
        margin-bottom: 20px !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2) !important;
    }
    h3 {
        font-size: 30px !important;
        font-weight: bold !important;
        color: #2980b9 !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
        border-bottom: 2px solid #3498db !important;
        padding-bottom: 5px !important;
    }
    .gradio-slider {
        font-size: 20px !important;
    }
    .gradio-radio {
        font-size: 20px !important;
    }
    .gradio-button {
        font-size: 24px !important;
        padding: 15px 30px !important;
        border-radius: 10px !important;
    }
    .gradio-textbox {
        font-size: 22px !important;
    }
    label {
        font-weight: bold !important;
        font-size: 22px !important;
        margin-bottom: 8px !important;
    }
    .section-container {
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        padding: 20px !important;
        margin-bottom: 15px !important;
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
    }
    .prediction-result {
        font-size: 26px !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 15px !important;
        border-radius: 10px !important;
        margin-top: 20px !important;
    }
    .high-risk {
        background-color: #ffcccc !important;
        color: #cc0000 !important;
    }
    .low-risk {
        background-color: #ccffcc !important;
        color: #006600 !important;
    }
    .bmi-info {
        font-size: 18px !important;
        margin-top: 10px !important;
        padding: 10px !important;
        background-color: #f0f0f0 !important;
        border-radius: 5px !important;
        border-left: 5px solid #3498db !important;
    }
    .feature-impact-plot {
        margin-top: 20px !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 15px rgba(0,0,0,0.3) !important;
        padding: 15px !important;
        background-color: #ffffff !important;
        border: 3px solid #3498db !important;
        transform: scale(1.05) !important;
        margin: 20px auto !important;
    }
    .plot-container {
        transform: scale(1.05) !important;
        margin: 20px auto !important;
    }
    .gradio-container .prose h1 {
        font-size: 48px !important;
    }
    .gradio-container .prose h3 {
        font-size: 32px !important;
    }
    .gradio-container .gradio-box {
        margin-bottom: 30px !important;
    }
    """
    
    with gr.Blocks(title="Heart Disease Risk Prediction", theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("# ❤️ Heart Disease Risk Prediction")
        gr.Markdown("### Enter patient information to predict heart disease risk.")
        
        with gr.Row():
            with gr.Column():
                # Demographics
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### Demographics")
                    age = gr.Slider(label="Age (in years)", minimum=20, maximum=100, value=55, step=1, scale=1)
                    sex = gr.Radio(label="Gender", choices=["Male", "Female"], value="Male", scale=1)
                
                # Physical measurements
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### Physical Measurements")
                    height = gr.Slider(label="Height (in cm)", minimum=140, maximum=210, value=170, step=1, scale=1)
                    weight = gr.Slider(label="Weight (in kg)", minimum=40, maximum=150, value=70, step=1, scale=1)
                    
                    # Display calculated BMI
                    def update_bmi(height, weight):
                        if height and weight:
                            bmi = calculate_bmi(height, weight)
                            category = get_bmi_category(bmi)
                            return f"BMI: {bmi:.1f} - {category}"
                        return "BMI will be calculated"
                    
                    bmi_display = gr.Markdown(elem_classes=["bmi-info"])
                    
                    # Update BMI when height or weight changes
                    height.change(update_bmi, [height, weight], bmi_display)
                    weight.change(update_bmi, [height, weight], bmi_display)
                
                # Chest pain type
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### Chest Pain Type")
                    cp = gr.Radio(
                        label="Chest pain type", 
                        choices=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                        value="Typical Angina",
                        scale=1
                    )
            
            with gr.Column():
                # Blood Tests
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### Blood Tests")
                    trestbps = gr.Slider(
                        label="Resting blood pressure (mm Hg)", 
                        minimum=80, 
                        maximum=200, 
                        value=120, 
                        step=1,
                        scale=1
                    )
                    chol = gr.Slider(
                        label="Serum cholesterol (mg/dl)", 
                        minimum=100, 
                        maximum=600, 
                        value=200, 
                        step=1,
                        scale=1
                    )
                    fbs = gr.Radio(
                        label="Fasting blood sugar > 120 mg/dl", 
                        choices=["Yes", "No"],
                        value="No",
                        scale=1
                    )
                
                # ECG Findings
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### ECG Findings")
                    restecg = gr.Radio(
                        label="Resting electrocardiographic results", 
                        choices=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                        value="Normal",
                        scale=1
                    )
                    thalach = gr.Slider(
                        label="Maximum heart rate achieved", 
                        minimum=60, 
                        maximum=220, 
                        value=150, 
                        step=1,
                        scale=1
                    )
        
        with gr.Row():
            with gr.Column():
                # Exercise Test
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### Exercise Test")
                    exang = gr.Radio(
                        label="Exercise induced angina", 
                        choices=["Yes", "No"],
                        value="No",
                        scale=1
                    )
                    oldpeak = gr.Slider(
                        label="ST depression induced by exercise relative to rest", 
                        minimum=0, 
                        maximum=6.2, 
                        value=0, 
                        step=0.1,
                        scale=1
                    )
                    slope = gr.Radio(
                        label="Slope of the peak exercise ST segment", 
                        choices=["Upsloping", "Flat", "Downsloping"],
                        value="Upsloping",
                        scale=1
                    )
            
            with gr.Column():
                # Additional Tests
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("### Additional Tests")
                    ca = gr.Radio(
                        label="Number of major vessels colored by fluoroscopy (0-3)", 
                        choices=["0", "1", "2", "3"],
                        value="0",
                        scale=1
                    )
                    thal = gr.Radio(
                        label="Thalassemia", 
                        choices=["Normal", "Fixed Defect", "Reversible Defect"],
                        value="Normal",
                        scale=1
                    )
        
        # Prediction button
        with gr.Row():
            predict_btn = gr.Button("SUBMIT", variant="primary", size="lg")
        
        # Results
        gr.Markdown("### Prediction Result")
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes=["section-container"]):
                    result_output = gr.Textbox(label="Prediction", scale=1, lines=1, max_lines=1, elem_classes=["prediction-result"])
                    risk_level_output = gr.Textbox(label="Risk Level", scale=1, lines=1, max_lines=1)
                    factors_output = gr.Textbox(label="Contributing Factors", lines=8, max_lines=12, scale=1)
            with gr.Column():
                with gr.Group(elem_classes=["section-container"]):
                    plot_output = gr.Plot(label="Feature Impact", scale=1, elem_classes=["feature-impact-plot"])
        
        # Map UI choices to model input values
        def preprocess_inputs(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, height, weight):
            # Map categorical variables
            sex_map = {"Male": 1, "Female": 0}
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            fbs_map = {"Yes": 1, "No": 0}
            restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            exang_map = {"Yes": 1, "No": 0}
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            
            return (
                age,
                sex_map[sex],
                cp_map[cp],
                trestbps,
                chol,
                fbs_map[fbs],
                restecg_map[restecg],
                thalach,
                exang_map[exang],
                oldpeak,
                slope_map[slope],
                int(ca),
                thal_map[thal],
                height,
                weight
            )
        
        # Post-process results to add styling
        def post_process_results(result, risk_level, factors, plot):
            # Add CSS class based on result
            if "High risk" in result:
                result = gr.update(value=result, elem_classes=["prediction-result", "high-risk"])
            else:
                result = gr.update(value=result, elem_classes=["prediction-result", "low-risk"])
            return result, risk_level, factors, plot
        
        # Connect button to prediction function
        predict_btn.click(
            fn=lambda *args: post_process_results(*predict_heart_disease(*preprocess_inputs(*args))),
            inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, height, weight],
            outputs=[result_output, risk_level_output, factors_output, plot_output]
        )
        
        # Initialize BMI display
        height.value = 170
        weight.value = 70
        bmi_display.value = update_bmi(170, 70)
    
    return interface

def open_browser():
    """Open browser after a delay to ensure server is up."""
    time.sleep(3)
    webbrowser.open('http://127.0.0.1:7860')

def main():
    """Main function to run the application."""
    global model
    
    print("Starting Heart Disease Prediction System...")
    print("Loading model...")
    model = ensure_model_exists()
    
    print("Starting web interface...")
    print("Creating public link (this may take a moment)...")
    
    # Create interface
    interface = create_interface()
    
    # Open browser automatically for local URL
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Launch with public link enabled
    interface.launch(share=True, show_error=True)
    
    print("\nPublic link has been created. Check the terminal output for the link.")
    print("You can share this link with others to let them use your application.")

if __name__ == "__main__":
    main() 