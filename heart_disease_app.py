"""
Heart Disease Prediction System - Executable Application
This script provides direct access to the Gradio web interface with optimized Feature Impact UI.
"""

import os
import sys
import webbrowser
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

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
    """Generate feature impact data with enhanced medical context and user-friendly explanations."""
    # Enhanced feature information with more detailed medical context
    feature_info = {
        'age': {
            'display': '🎂 Age',
            'info': 'Heart disease risk doubles every decade after 45. Regular check-ups become crucial.',
            'normal_range': '20-45 years',
            'high_risk': '>65 years',
            'category': 'Demographics',
            'severity_levels': {'low': '<45', 'medium': '45-65', 'high': '>65'}
        },
        'sex': {
            'display': '👥 Gender',
            'info': 'Men have 2-3x higher risk than women before menopause due to hormonal protection.',
            'normal_range': 'Female (lower risk)',
            'high_risk': 'Male (higher risk)',
            'category': 'Demographics',
            'severity_levels': {'low': 'Female', 'high': 'Male'}
        },
        'cp': {
            'display': '💓 Chest Pain Pattern',
            'info': 'Typical angina strongly suggests coronary artery disease requiring immediate attention.',
            'normal_range': 'No chest pain',
            'high_risk': 'Typical angina',
            'category': 'Symptoms',
            'severity_levels': {'low': 'Asymptomatic', 'medium': 'Atypical', 'high': 'Typical angina'}
        },
        'trestbps': {
            'display': '🩺 Resting Blood Pressure',
            'info': 'Each 20mmHg increase doubles cardiovascular risk. Control is essential.',
            'normal_range': '90-120 mmHg',
            'high_risk': '>140 mmHg',
            'category': 'Vital Signs',
            'severity_levels': {'low': '<120', 'medium': '120-140', 'high': '>140'}
        },
        'chol': {
            'display': '🧪 Total Cholesterol',
            'info': '23% increased risk per 40mg/dL above 200. Diet and medication can help.',
            'normal_range': '<200 mg/dL',
            'high_risk': '>240 mg/dL',
            'category': 'Blood Tests',
            'severity_levels': {'low': '<200', 'medium': '200-240', 'high': '>240'}
        },
        'fbs': {
            'display': '🍯 Fasting Blood Sugar',
            'info': 'Diabetes doubles heart disease risk. Blood sugar control is vital.',
            'normal_range': '<100 mg/dL',
            'high_risk': '>126 mg/dL',
            'category': 'Blood Tests',
            'severity_levels': {'low': '<100', 'medium': '100-126', 'high': '>126'}
        },
        'restecg': {
            'display': '📈 Resting ECG',
            'info': 'Abnormal patterns indicate electrical problems that increase risk 5-fold.',
            'normal_range': 'Normal',
            'high_risk': 'ST-T abnormality',
            'category': 'Heart Tests',
            'severity_levels': {'low': 'Normal', 'medium': 'Minor changes', 'high': 'ST-T abnormality'}
        },
        'thalach': {
            'display': '💗 Maximum Heart Rate',
            'info': 'Lower maximum heart rate may indicate decreased cardiac function.',
            'normal_range': '>150 bpm',
            'high_risk': '<120 bpm',
            'category': 'Exercise Tests',
            'severity_levels': {'high': '<120', 'medium': '120-150', 'low': '>150'}
        },
        'exang': {
            'display': '🏃 Exercise-Induced Chest Pain',
            'info': 'Chest pain during exercise indicates poor blood flow, 3x higher risk.',
            'normal_range': 'No pain',
            'high_risk': 'Pain present',
            'category': 'Exercise Tests',
            'severity_levels': {'low': 'No', 'high': 'Yes'}
        },
        'oldpeak': {
            'display': '📊 ST Depression (Exercise)',
            'info': 'Values >2mm indicate severe ischemia requiring immediate attention.',
            'normal_range': '<1mm',
            'high_risk': '>2mm',
            'category': 'Exercise Tests',
            'severity_levels': {'low': '<1', 'medium': '1-2', 'high': '>2'}
        },
        'slope': {
            'display': '📉 Exercise ST Slope',
            'info': 'Downsloping pattern indicates poor cardiac response to exercise.',
            'normal_range': 'Upsloping',
            'high_risk': 'Downsloping',
            'category': 'Exercise Tests',
            'severity_levels': {'low': 'Upsloping', 'medium': 'Flat', 'high': 'Downsloping'}
        },
        'ca': {
            'display': '🔍 Blocked Arteries',
            'info': 'Risk increases 2x per major vessel with significant blockage.',
            'normal_range': '0 vessels',
            'high_risk': '≥2 vessels',
            'category': 'Imaging',
            'severity_levels': {'low': '0', 'medium': '1', 'high': '≥2'}
        },
        'thal': {
            'display': '🫀 Heart Muscle Perfusion',
            'info': 'Reversible defects indicate areas of poor blood flow, 3x higher risk.',
            'normal_range': 'Normal',
            'high_risk': 'Reversible defect',
            'category': 'Imaging',
            'severity_levels': {'low': 'Normal', 'medium': 'Fixed defect', 'high': 'Reversible defect'}
        }
    }
    
    # Get feature importances from the model's explanation
    explanation = model.explain_prediction(input_data)
    
    # Extract contributing factors with enhanced information
    impacts = []
    for factor in explanation['contributing_factors']:
        feature = factor['feature']
        if feature in feature_info:
            info = feature_info[feature]
            
            # Determine severity level based on value
            severity = 'medium'  # default
            if 'severity_levels' in info:
                value = factor['value']
                levels = info['severity_levels']
                # Simple severity determination logic
                if 'high' in levels:
                    high_threshold = levels.get('high', '')
                    if isinstance(value, (int, float)):
                        if '>' in str(high_threshold):
                            threshold = float(str(high_threshold).replace('>', ''))
                            if value > threshold:
                                severity = 'high'
                        elif '<' in str(high_threshold):
                            threshold = float(str(high_threshold).replace('<', ''))
                            if value < threshold:
                                severity = 'high'
                    elif str(value) in str(high_threshold):
                        severity = 'high'
            
            impacts.append({
                'name': info['display'],
                'impact': factor['importance'],
                'info': info['info'],
                'value': factor['value'],
                'normal_range': info['normal_range'],
                'high_risk': info['high_risk'],
                'category': info['category'],
                'severity': severity,
                'raw_feature': feature
            })
    
    # Add BMI impact with enhanced information
    bmi_found = any('BMI' in item['name'] for item in impacts)
    if not bmi_found:
        bmi_value = calculate_bmi(input_data.get('height', 170), input_data.get('weight', 70))
        bmi_category = get_bmi_category(bmi_value)
        
        # Determine BMI impact and severity
        bmi_impact = 0.05  # Base impact
        severity = 'low'
        
        if bmi_value >= 30:  # Obese
            bmi_impact = 0.18
            severity = 'high'
        elif bmi_value >= 25:  # Overweight
            bmi_impact = 0.12
            severity = 'medium'
        elif bmi_value < 18.5:  # Underweight
            bmi_impact = 0.08
            severity = 'medium'
            
        impacts.append({
            'name': '⚖️ Body Mass Index',
            'impact': bmi_impact,
            'info': f'BMI >30 increases heart disease risk by 50%. Current: {bmi_category}',
            'value': bmi_value,
            'normal_range': '18.5-25',
            'high_risk': '>30',
            'category': 'Physical',
            'severity': severity,
            'raw_feature': 'bmi'
        })
    
    return impacts

def create_enhanced_visualization(feature_impacts, explanation, input_data, bmi):
    """Create an enhanced, more user-friendly feature impact visualization."""
    # Set up the figure with better styling
    plt.style.use('default')  # Start fresh
    fig = plt.figure(figsize=(20, 18), dpi=120, facecolor='white')
    
    # Create a sophisticated layout
    gs = plt.GridSpec(4, 2, height_ratios=[0.5, 3, 1, 0.8], width_ratios=[3, 1], 
                     hspace=0.3, wspace=0.2, left=0.08, right=0.92, top=0.95, bottom=0.05)
    
    # Main title section
    title_ax = plt.subplot(gs[0, :])
    title_ax.axis('off')
    
    # Add gradient background for title
    title_gradient = np.linspace(0, 1, 100).reshape(1, -1)
    title_ax.imshow(title_gradient, aspect='auto', cmap='Blues', alpha=0.3, extent=[0, 1, 0, 1])
    
    # Main title with better styling
    title_ax.text(0.5, 0.7, '❤️ PERSONALIZED HEART HEALTH ANALYSIS', 
                 ha='center', va='center', fontsize=44, fontweight='bold', 
                 color='#1a365d', transform=title_ax.transAxes, fontfamily='sans-serif')
    
    # Subtitle with patient info
    bmi_category = get_bmi_category(bmi)
    risk_level = explanation['risk_level']
    probability = explanation['probability']
    
    subtitle = f"Risk Level: {risk_level} | Probability: {probability:.1%} | BMI: {bmi:.1f} ({bmi_category})"
    title_ax.text(0.5, 0.2, subtitle, ha='center', va='center', fontsize=28, 
                 color='#2c5282', transform=title_ax.transAxes, fontweight='600')
    
    # Main feature impact plot
    main_ax = plt.subplot(gs[1, :])
    
    # Sort features by impact for better visualization
    sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x['impact']), reverse=True)[:10]
    
    # Prepare data for plotting
    names = [item['name'] for item in sorted_impacts]
    impacts = [item['impact'] for item in sorted_impacts]
    categories = [item['category'] for item in sorted_impacts]
    severities = [item['severity'] for item in sorted_impacts]
    
    # Create a color map based on categories and severities
    category_colors = {
        'Demographics': '#6366f1',
        'Symptoms': '#ef4444', 
        'Vital Signs': '#f59e0b',
        'Blood Tests': '#10b981',
        'Heart Tests': '#8b5cf6',
        'Exercise Tests': '#06b6d4',
        'Imaging': '#f97316',
        'Physical': '#84cc16'
    }
    
    # Adjust colors based on severity
    bar_colors = []
    for category, severity in zip(categories, severities):
        base_color = category_colors.get(category, '#6b7280')
        if severity == 'high':
            bar_colors.append(base_color)
        elif severity == 'medium':
            # Lighter version
            from matplotlib.colors import to_rgba, to_hex
            rgba = to_rgba(base_color)
            lighter = tuple(min(1.0, c + 0.2) for c in rgba[:3]) + (rgba[3],)
            bar_colors.append(lighter)
        else:
            # Much lighter version
            from matplotlib.colors import to_rgba, to_hex
            rgba = to_rgba(base_color)
            much_lighter = tuple(min(1.0, c + 0.4) for c in rgba[:3]) + (rgba[3],)
            bar_colors.append(much_lighter)
    
    # Create horizontal bar chart with enhanced styling
    y_positions = range(len(names))
    bars = main_ax.barh(y_positions, impacts, color=bar_colors, height=0.7,
                       edgecolor='#374151', linewidth=1.5, alpha=0.9)
    
    # Add value labels with better positioning
    for i, (bar, impact) in enumerate(zip(bars, impacts)):
        width = bar.get_width()
        label_x = width + (0.002 if width >= 0 else -0.005)
        
        # Color based on impact direction
        text_color = '#dc2626' if width > 0 else '#059669'
        
        main_ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center', ha='left' if width >= 0 else 'right',
                    fontsize=20, fontweight='bold', color=text_color,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=text_color, alpha=0.8))
    
    # Customize the main plot
    main_ax.set_yticks(y_positions)
    main_ax.set_yticklabels(names, fontsize=22, fontweight='600')
    main_ax.set_xlabel('Impact on Heart Disease Risk', fontsize=28, fontweight='bold', 
                      color='#374151', labelpad=15)
    main_ax.set_title('KEY FACTORS AFFECTING YOUR HEART HEALTH', 
                     fontsize=36, fontweight='bold', color='#1f2937', pad=25)
    
    # Add reference line at x=0
    main_ax.axvline(x=0, color='#6b7280', linestyle='-', alpha=0.7, linewidth=2)
    
    # Enhanced grid
    main_ax.grid(True, axis='x', linestyle='--', alpha=0.4, color='#9ca3af')
    main_ax.set_facecolor('white')
    
    # Remove top and right spines
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['left'].set_linewidth(2)
    main_ax.spines['bottom'].set_linewidth(2)
    
    # Risk gauge in the bottom left
    gauge_ax = plt.subplot(gs[2, 0])
    
    # Create risk gauge
    risk_score = probability * 100
    gauge_colors = ['#059669', '#fbbf24', '#f59e0b', '#ef4444', '#dc2626']
    gauge_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    gauge_ranges = [0, 20, 40, 60, 80, 100]
    
    # Create gauge background
    for i in range(len(gauge_colors)):
        start, end = gauge_ranges[i], gauge_ranges[i+1]
        gauge_ax.barh(0, end-start, left=start, height=0.5, 
                     color=gauge_colors[i], alpha=0.7, edgecolor='white', linewidth=2)
    
    # Add risk pointer
    pointer_color = gauge_colors[min(4, int(risk_score // 20))]
    gauge_ax.scatter(risk_score, 0, s=1000, marker='v', color=pointer_color, 
                    edgecolor='#374151', linewidth=3, zorder=10)
    
    # Add percentage text
    gauge_ax.text(risk_score, -0.3, f'{risk_score:.1f}%', ha='center', va='top',
                 fontsize=24, fontweight='bold', color=pointer_color)
    
    gauge_ax.set_xlim(0, 100)
    gauge_ax.set_ylim(-0.5, 0.5)
    gauge_ax.set_xticks([0, 20, 40, 60, 80, 100])
    gauge_ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=18)
    gauge_ax.set_yticks([])
    gauge_ax.set_title('RISK PROBABILITY GAUGE', fontsize=24, fontweight='bold', 
                      color='#374151', pad=15)
    
    # Category legend in bottom right
    legend_ax = plt.subplot(gs[2, 1])
    legend_ax.axis('off')
    
    # Create category legend
    unique_categories = list(set(categories))
    legend_elements = []
    for cat in unique_categories:
        color = category_colors.get(cat, '#6b7280')
        legend_elements.append(mpatches.Patch(color=color, label=cat))
    
    legend = legend_ax.legend(handles=legend_elements, loc='center', fontsize=16,
                             title='Risk Factor Categories', title_fontsize=18,
                             frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#d1d5db')
    
    # Recommendations section
    rec_ax = plt.subplot(gs[3, :])
    rec_ax.axis('off')
    
    # Add recommendations background
    rec_bg = FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
                           facecolor='#eff6ff', edgecolor='#3b82f6', linewidth=2,
                           transform=rec_ax.transAxes)
    rec_ax.add_patch(rec_bg)
    
    # Generate personalized recommendations
    recommendations = generate_personalized_recommendations(sorted_impacts, bmi, risk_score)
    
    rec_ax.text(0.5, 0.75, '🎯 PERSONALIZED RECOMMENDATIONS', ha='center', va='center',
               fontsize=28, fontweight='bold', color='#1e40af', transform=rec_ax.transAxes)
    
    # Display recommendations in columns
    rec_text = ' | '.join([f"• {rec}" for rec in recommendations[:4]])
    rec_ax.text(0.5, 0.35, rec_text, ha='center', va='center', fontsize=18,
               color='#374151', transform=rec_ax.transAxes, fontweight='500',
               wrap=True)
    
    return fig

def create_enhanced_feature_plot(feature_impacts, explanation, input_data, bmi):
    """Create an enhanced, more user-friendly feature impact visualization."""
    # Set up modern styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(22, 20), dpi=120, facecolor='white')
    
    # Create sophisticated layout with better proportions
    gs = plt.GridSpec(4, 2, height_ratios=[0.4, 3.2, 1.2, 0.8], width_ratios=[2.5, 1], 
                     hspace=0.35, wspace=0.25, left=0.06, right=0.94, top=0.96, bottom=0.04)
    
    # Enhanced title section with gradient background
    title_ax = plt.subplot(gs[0, :])
    title_ax.axis('off')
    
    # Create gradient background for title
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    title_ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.2, extent=[0, 1, 0, 1])
    
    # Main title with medical icons
    title_ax.text(0.5, 0.75, '🫀 COMPREHENSIVE HEART HEALTH ANALYSIS', 
                 ha='center', va='center', fontsize=48, fontweight='bold', 
                 color='#1a365d', transform=title_ax.transAxes, fontfamily='Arial')
    
    # Enhanced subtitle with patient data
    bmi_category = get_bmi_category(bmi)
    risk_level = explanation['risk_level']
    probability = explanation['probability']
    
    # Color-coded risk level
    risk_colors = {'Low Risk': '#10b981', 'Moderate Risk': '#f59e0b', 
                   'High Risk': '#ef4444', 'Very High Risk': '#dc2626'}
    risk_color = risk_colors.get(risk_level, '#6b7280')
    
    subtitle = f"Risk Level: {risk_level} ({probability:.1%}) | BMI: {bmi:.1f} ({bmi_category})"
    title_ax.text(0.5, 0.25, subtitle, ha='center', va='center', fontsize=32, 
                 color=risk_color, transform=title_ax.transAxes, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=risk_color, alpha=0.9))
    
    # Main feature impact visualization
    main_ax = plt.subplot(gs[1, :])
    
    # Sort and prepare data
    sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x['impact']), reverse=True)[:12]
    
    names = [item['name'] for item in sorted_impacts]
    impacts = [item['impact'] for item in sorted_impacts]
    categories = [item['category'] for item in sorted_impacts]
    severities = [item['severity'] for item in sorted_impacts]
    
    # Enhanced color scheme based on categories and severities
    category_colors = {
        'Demographics': '#8b5cf6', 'Symptoms': '#ef4444', 'Vital Signs': '#f59e0b',
        'Blood Tests': '#10b981', 'Heart Tests': '#3b82f6', 'Exercise Tests': '#06b6d4',
        'Imaging': '#f97316', 'Physical': '#84cc16'
    }
    
    # Create bars with enhanced visual effects
    y_positions = np.arange(len(names))
    
    # Create main bars with gradient effect
    bars = []
    for i, (impact, category, severity) in enumerate(zip(impacts, categories, severities)):
        base_color = category_colors.get(category, '#6b7280')
        
        # Adjust alpha based on severity
        alpha = 0.9 if severity == 'high' else 0.7 if severity == 'medium' else 0.5
        
        bar = main_ax.barh(y_positions[i], impact, height=0.75, 
                          color=base_color, alpha=alpha, 
                          edgecolor='#374151', linewidth=1.5)
        bars.append(bar)
        
        # Add severity indicator
        severity_colors = {'high': '#dc2626', 'medium': '#f59e0b', 'low': '#10b981'}
        severity_marker = severity_colors.get(severity, '#6b7280')
        
        # Add small severity indicator dot
        main_ax.scatter(impact + 0.003, y_positions[i], s=200, 
                       color=severity_marker, edgecolor='white', linewidth=2, zorder=10)
    
    # Enhanced value labels with better styling
    for i, (bar, impact, severity) in enumerate(zip(bars, impacts, severities)):
        width = bar[0].get_width()
        label_x = width + 0.005 if width >= 0 else width - 0.008
        
        # Choose text color based on impact direction and severity
        if width > 0:
            text_color = '#dc2626' if severity == 'high' else '#ef4444'
            bg_color = '#fef2f2'
        else:
            text_color = '#059669' if severity == 'high' else '#10b981'
            bg_color = '#f0fdf4'
        
        main_ax.text(label_x, bar[0].get_y() + bar[0].get_height()/2, 
                    f'{width:.3f}', va='center', ha='left' if width >= 0 else 'right',
                    fontsize=22, fontweight='bold', color=text_color,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=bg_color, 
                             edgecolor=text_color, alpha=0.8, linewidth=1.5))
    
    # Enhanced axis formatting
    main_ax.set_yticks(y_positions)
    main_ax.set_yticklabels(names, fontsize=24, fontweight='600', color='#374151')
    main_ax.set_xlabel('Impact on Heart Disease Risk', fontsize=32, fontweight='bold', 
                      color='#1f2937', labelpad=20)
    main_ax.set_title('🎯 KEY RISK FACTORS & THEIR IMPACT', 
                     fontsize=40, fontweight='bold', color='#1f2937', pad=30)
    
    # Enhanced reference line and grid
    main_ax.axvline(x=0, color='#374151', linestyle='-', alpha=0.8, linewidth=3)
    main_ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='#9ca3af', linewidth=1)
    main_ax.set_facecolor('white')
    
    # Style spines
    for spine in main_ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#d1d5db')
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    
    # Enhanced risk gauge with multiple indicators
    gauge_ax = plt.subplot(gs[2, 0])
    
    # Create segmented risk gauge
    risk_score = probability * 100
    segments = [(0, 20, '#10b981', 'Very Low'), (20, 40, '#84cc16', 'Low'), 
                (40, 60, '#f59e0b', 'Moderate'), (60, 80, '#ef4444', 'High'), 
                (80, 100, '#dc2626', 'Very High')]
    
    for start, end, color, label in segments:
        gauge_ax.barh(0, end-start, left=start, height=0.6, 
                     color=color, alpha=0.8, edgecolor='white', linewidth=3)
        # Add segment labels
        gauge_ax.text(start + (end-start)/2, -0.15, label, ha='center', va='top',
                     fontsize=16, fontweight='bold', color=color)
    
    # Enhanced risk pointer with animation-style effect
    pointer_size = 1500
    pointer_color = segments[min(4, int(risk_score // 20))][2]
    
    # Add glow effect around pointer
    for size, alpha in [(pointer_size*1.5, 0.3), (pointer_size*1.2, 0.5), (pointer_size, 1.0)]:
        gauge_ax.scatter(risk_score, 0, s=size, marker='v', color=pointer_color, 
                        alpha=alpha, edgecolor='white', linewidth=2, zorder=10-alpha*10)
    
    # Risk percentage with enhanced styling
    gauge_ax.text(risk_score, -0.45, f'{risk_score:.1f}%', ha='center', va='top',
                 fontsize=32, fontweight='bold', color=pointer_color,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                          edgecolor=pointer_color, linewidth=2, alpha=0.95))
    
    gauge_ax.set_xlim(0, 100)
    gauge_ax.set_ylim(-0.6, 0.4)
    gauge_ax.set_xticks([0, 20, 40, 60, 80, 100])
    gauge_ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=20)
    gauge_ax.set_yticks([])
    gauge_ax.set_title('🎯 RISK PROBABILITY GAUGE', fontsize=28, fontweight='bold', 
                      color='#374151', pad=20)
    gauge_ax.set_facecolor('white')
    
    # Enhanced category legend
    legend_ax = plt.subplot(gs[2, 1])
    legend_ax.axis('off')
    
    # Create interactive-style legend
    unique_categories = sorted(list(set(categories)))
    legend_elements = []
    for i, cat in enumerate(unique_categories):
        color = category_colors.get(cat, '#6b7280')
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                           edgecolor='white', linewidth=2, label=cat))
    
    legend = legend_ax.legend(handles=legend_elements, loc='center', fontsize=18,
                             title='📊 Risk Categories', title_fontsize=22,
                             frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#d1d5db')
    legend.get_frame().set_linewidth(2)
    
    # Enhanced recommendations section
    rec_ax = plt.subplot(gs[3, :])
    rec_ax.axis('off')
    
    # Create modern recommendations panel
    rec_bg = plt.Rectangle((0.01, 0.05), 0.98, 0.9, facecolor='#eff6ff', 
                          edgecolor='#3b82f6', linewidth=3, alpha=0.9,
                          transform=rec_ax.transAxes)
    rec_ax.add_patch(rec_bg)
    
    # Generate and display recommendations
    recommendations = generate_personalized_recommendations(sorted_impacts, bmi, risk_score)
    
    rec_ax.text(0.5, 0.8, '💡 PERSONALIZED ACTION PLAN', ha='center', va='center',
               fontsize=32, fontweight='bold', color='#1e40af', 
               transform=rec_ax.transAxes)
    
    # Display recommendations with icons
    icons = ['🏥', '💊', '🏃‍♂️', '🥗', '📊', '🧘‍♂️']
    for i, rec in enumerate(recommendations[:4]):
        x_pos = 0.125 + (i * 0.22)
        icon = icons[i] if i < len(icons) else '✓'
        
        # Icon with background
        rec_ax.text(x_pos, 0.5, icon, ha='center', va='center', fontsize=28,
                   transform=rec_ax.transAxes,
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', 
                            edgecolor='#3b82f6', linewidth=2))
        
        # Recommendation text
        rec_ax.text(x_pos, 0.25, rec, ha='center', va='center', fontsize=16,
                   color='#1e40af', fontweight='bold', transform=rec_ax.transAxes,
                   wrap=True)
    
    # Add medical disclaimer
    rec_ax.text(0.5, 0.05, '⚠️ Always consult with your healthcare provider before making health decisions',
               ha='center', va='center', fontsize=14, color='#6b7280', style='italic',
               transform=rec_ax.transAxes)
    
    return fig

def generate_personalized_recommendations(impacts, bmi, risk_score):
    """Generate personalized recommendations based on risk factors."""
    recommendations = []
    
    # Check top risk factors
    top_factors = impacts[:3]  # Top 3 factors
    
    for factor in top_factors:
        feature = factor['raw_feature']
        severity = factor['severity']
        
        if severity in ['high', 'medium']:
            if 'age' in feature:
                recommendations.append("Regular cardiac check-ups")
            elif 'cp' in feature:
                recommendations.append("Consult cardiologist for chest pain")
            elif 'trestbps' in feature:
                recommendations.append("Monitor blood pressure daily")
            elif 'chol' in feature:
                recommendations.append("Heart-healthy diet & statins")
            elif 'bmi' in feature:
                if bmi >= 30:
                    recommendations.append("Weight loss program")
                elif bmi >= 25:
                    recommendations.append("Moderate weight reduction")
            elif 'thalach' in feature:
                recommendations.append("Gradual exercise program")
            elif 'exang' in feature:
                recommendations.append("Supervised exercise testing")
    
    # Add general recommendations
    if len(recommendations) < 3:
        general_recs = [
            "150 min/week moderate exercise",
            "Mediterranean diet",
            "Stress management",
            "Quit smoking if applicable",
            "Limit alcohol intake"
        ]
        recommendations.extend(general_recs)
    
    return recommendations[:4]

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
    
    # Create enhanced feature impact visualization
    fig = create_enhanced_feature_plot(feature_impacts, explanation, input_data, bmi)
    
    return result, risk_level, contributing_factors, fig

def create_interface():
    """Create the Gradio interface with larger text and better visibility."""
    # Enhanced CSS for modern, user-friendly interface
    custom_css = """
    .gradio-container {
        font-size: 22px !important;
        max-width: 1400px !important;
        margin: auto !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh !important;
        padding: 20px !important;
    }
    h1 {
        font-size: 52px !important;
        font-weight: bold !important;
        background: linear-gradient(45deg, #e74c3c, #f39c12) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-bottom: 25px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important;
        padding: 15px !important;
    }
    h3 {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
        border-bottom: 3px solid #3498db !important;
        padding-bottom: 8px !important;
        position: relative !important;
    }
    h3::before {
        content: '🏥' !important;
        margin-right: 10px !important;
    }
    .gradio-slider {
        font-size: 20px !important;
        transition: all 0.3s ease !important;
    }
    .gradio-slider:hover {
        transform: scale(1.02) !important;
    }
    .gradio-radio {
        font-size: 20px !important;
        transition: all 0.3s ease !important;
    }
    .gradio-radio:hover {
        transform: scale(1.02) !important;
    }
    .gradio-button {
        font-size: 28px !important;
        padding: 20px 40px !important;
        border-radius: 15px !important;
        background: linear-gradient(45deg, #3498db, #2ecc71) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: bold !important;
    }
    .gradio-button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 24px rgba(0,0,0,0.3) !important;
        background: linear-gradient(45deg, #2980b9, #27ae60) !important;
    }
    .gradio-textbox {
        font-size: 22px !important;
        border-radius: 10px !important;
        border: 2px solid #bdc3c7 !important;
        transition: all 0.3s ease !important;
    }
    .gradio-textbox:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 10px rgba(52, 152, 219, 0.3) !important;
    }
    label {
        font-weight: bold !important;
        font-size: 22px !important;
        margin-bottom: 8px !important;
        color: #2c3e50 !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5) !important;
    }
    .section-container {
        border-radius: 20px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
        padding: 25px !important;
        margin-bottom: 20px !important;
        background: rgba(255,255,255,0.95) !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    .section-container:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2) !important;
    }
    .prediction-result {
        font-size: 30px !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 20px !important;
        border-radius: 15px !important;
        margin-top: 25px !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .high-risk {
        background: linear-gradient(135deg, #ffcccc, #ffaaaa) !important;
        color: #cc0000 !important;
        border: 3px solid #e74c3c !important;
    }
    .low-risk {
        background: linear-gradient(135deg, #ccffcc, #aaffaa) !important;
        color: #006600 !important;
        border: 3px solid #27ae60 !important;
    }
    .bmi-info {
        font-size: 20px !important;
        margin-top: 15px !important;
        padding: 15px !important;
        background: linear-gradient(135deg, #e8f4fd, #d6eaff) !important;
        border-radius: 10px !important;
        border-left: 5px solid #3498db !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .bmi-info:hover {
        transform: scale(1.02) !important;
    }
    .feature-impact-plot {
        margin-top: 25px !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 16px 32px rgba(0,0,0,0.2) !important;
        padding: 20px !important;
        background: rgba(255,255,255,0.98) !important;
        border: 4px solid #3498db !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.5s ease !important;
    }
    .feature-impact-plot:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3) !important;
    }
    .plot-container {
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    .gradio-container .prose h1 {
        font-size: 54px !important;
    }
    .gradio-container .prose h3 {
        font-size: 32px !important;
    }
    .gradio-container .gradio-box {
        margin-bottom: 25px !important;
        border-radius: 15px !important;
    }
    /* Enhanced animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .gradio-button:active {
        animation: pulse 0.3s ease-in-out !important;
    }
    /* Better mobile responsiveness */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
            font-size: 18px !important;
        }
        h1 {
            font-size: 36px !important;
        }
        .gradio-button {
            font-size: 22px !important;
            padding: 15px 25px !important;
        }
        .section-container {
            padding: 15px !important;
        }
    }
    """
    
    with gr.Blocks(title="AI-Powered Heart Health Analysis", theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("# 💓 AI-Powered Heart Health Analysis")
        gr.Markdown("### 🔬 Advanced Feature Impact Analysis • Personalized Risk Assessment • Medical-Grade Insights")
        gr.Markdown("---")
        
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
        
        # Enhanced prediction button
        with gr.Row():
            gr.Markdown("---")
        with gr.Row():
            predict_btn = gr.Button("🔬 ANALYZE MY HEART HEALTH", variant="primary", size="lg")
        
        # Enhanced Results Section
        gr.Markdown("### 📊 Comprehensive Risk Analysis Results")
        gr.Markdown("*Complete the form above and click **ANALYZE** to generate your personalized heart health report*")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("#### 🎯 Risk Assessment")
                    result_output = gr.Textbox(label="Prediction Result", scale=1, lines=2, max_lines=3, elem_classes=["prediction-result"])
                    risk_level_output = gr.Textbox(label="Risk Category", scale=1, lines=1, max_lines=1)
                    
                    gr.Markdown("#### 📋 Key Contributing Factors")
                    factors_output = gr.Textbox(label="Detailed Analysis", lines=10, max_lines=15, scale=1)
                    
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["section-container"]):
                    gr.Markdown("#### 📈 Interactive Feature Impact Visualization")
                    plot_output = gr.Plot(label="Advanced Analytics Dashboard", scale=1, elem_classes=["feature-impact-plot"])
        
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
        
        # Educational Information Section
        with gr.Row():
            gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 💡 Understanding Your Results
                
                **Feature Impact Analysis**: Shows how each factor contributes to your overall risk
                - **Red bars**: Factors that increase your risk
                - **Green bars**: Factors that decrease your risk  
                - **Bar length**: Indicates the strength of impact
                
                **Risk Categories**:
                - 🟢 **Low Risk** (0-20%): Maintain healthy lifestyle
                - 🟡 **Moderate Risk** (20-60%): Consider lifestyle changes
                - 🔴 **High Risk** (60%+): Seek medical consultation
                """)
            with gr.Column():
                gr.Markdown("""
                ### 🏥 Medical Disclaimer
                
                **Important**: This tool is for educational purposes only and should not replace professional medical advice.
                
                **Always consult with a qualified healthcare provider** for:
                - Medical diagnosis and treatment
                - Interpretation of test results  
                - Personalized medical recommendations
                - Emergency medical situations
                
                **Data Privacy**: Your information is processed locally and not stored.
                """)
        
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