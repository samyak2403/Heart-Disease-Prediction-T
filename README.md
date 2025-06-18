# Heart Disease Prediction System 🫀

A comprehensive machine learning system for predicting heart disease risk using patient data. This project provides multiple interfaces (web, console, and executable) with advanced visualization and explanation capabilities.

## 🌟 Features

### 🔬 **Core Prediction Engine**
- **Random Forest Classifier** with hyperparameter optimization
- **Feature Importance Analysis** to identify key risk factors
- **Personalized Risk Assessment** with detailed explanations
- **Real-time Probability Scoring** (0-100% risk scale)
- **Medical Context Integration** with BMI calculations

### 🎨 **Advanced Visualizations**
- **Interactive Risk Gauge** with color-coded severity levels
- **Feature Impact Charts** showing positive/negative contributions
- **Category-based Risk Analysis** (Demographics, Symptoms, Vital Signs, etc.)
- **Personalized Recommendations** based on individual risk factors
- **Professional Medical-style Reports** with clean white backgrounds

### 🖥️ **Multiple User Interfaces**

#### 1. **Web Interface (Gradio)**
- Modern, responsive web UI accessible via browser
- Real-time predictions with interactive controls
- Enhanced visualization dashboard
- Mobile-friendly design
- Automatic BMI calculation

#### 2. **Console Interface**
- Command-line interface for quick predictions
- Step-by-step input guidance
- Detailed result explanations
- Perfect for automation and scripting

#### 3. **Standalone Executable**
- PyInstaller-built Windows executable
- No Python installation required
- Portable distribution
- Complete self-contained application

### 📊 **Comprehensive Feature Set**

#### **Demographic Features**
- Age (20-100 years)
- Gender (Male/Female)
- Weight and Height (with automatic BMI calculation)

#### **Medical History**
- Chest Pain Type (4 categories: Typical angina, Atypical angina, Non-anginal pain, Asymptomatic)
- Resting Blood Pressure (80-200 mmHg)
- Serum Cholesterol (100-600 mg/dl)
- Fasting Blood Sugar (>120 mg/dl indicator)
- Family History of Heart Disease

#### **ECG & Heart Tests**
- Resting Electrocardiographic Results (Normal, ST-T abnormality, Left ventricular hypertrophy)
- Maximum Heart Rate Achieved (60-220 bpm)
- Exercise-Induced Angina (Yes/No)

#### **Exercise Test Results**
- ST Depression (0-10 range)
- Peak Exercise ST Segment Slope (Upsloping, Flat, Downsloping)

#### **Advanced Diagnostics**
- Number of Major Vessels (0-3) colored by fluoroscopy
- Thalassemia Test Results (Normal, Fixed defect, Reversible defect)

### 🧠 **Machine Learning Capabilities**

#### **Model Training & Evaluation**
- Automated data preprocessing with feature scaling
- Cross-validation with GridSearchCV
- Comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Score
  - Confusion Matrix
  - Classification Report

#### **Feature Engineering**
- Automatic BMI calculation from height/weight
- Feature importance ranking
- Categorical encoding (One-hot encoding)
- Numerical feature standardization

#### **Model Persistence**
- Joblib-based model serialization
- Automatic model saving/loading
- Preprocessor state preservation
- Cross-session compatibility

### 📈 **Analysis & Reporting Tools**

#### **Feature Importance Analysis**
- Visual feature importance plots
- Top contributing factors identification
- Category-wise risk breakdown
- Medical context explanations

#### **Personalized Recommendations**
- Risk-specific lifestyle suggestions
- Medical consultation recommendations
- Exercise and diet guidance
- Monitoring suggestions based on risk factors

#### **Risk Level Classification**
- **Low Risk** (<20%): Minimal intervention needed
- **Moderate Risk** (20-50%): Lifestyle modifications recommended
- **High Risk** (50-80%): Medical consultation advised
- **Very High Risk** (>80%): Immediate medical attention suggested

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
   cd heart-disease-prediction
   ```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

#### Option A: Web Interface (Recommended)
```bash
   python heart_disease_app.py
   ```
Access the web interface at `http://localhost:7860`

#### Option B: Command Line Interface
```bash
python main.py
```

#### Option C: Feature Analysis with Gradio
```bash
python analyze_features.py
```

#### Option D: Feature-Only Analysis
```bash
python analyze_features_only.py  # Generate feature importance plots without UI
```

#### Option E: Simple App Version
```bash
python simple_heart_app.py  # Minimal version for basic predictions
```

### 📦 Building Standalone Executable

Create a portable Windows executable:

```bash
   python build_exe.py
   ```

Or use the batch file:
```bash
build_exe.bat
```

The executable will be created in the `dist/` folder.

#### Quick Launch Options
For Windows users, you can also use:
```bash
run_heart_disease_app.bat  # Quick launcher for web interface
```

## 📋 Usage Examples

### Web Interface Usage
1. Launch the application: `python heart_disease_app.py`
2. Open browser to the provided URL
3. Fill in patient information using the intuitive sliders and dropdowns
4. Click "Predict Heart Disease Risk"
5. View comprehensive results with visualizations and recommendations

### Console Interface Usage
```python
from src.ui.console_interface import ConsoleInterface

interface = ConsoleInterface('models/heart_disease_model.joblib')
interface.run()
```

### Programmatic Usage
```python
from src.models.model import HeartDiseaseModel

# Load trained model
model = HeartDiseaseModel.load('models/heart_disease_model.joblib')

# Make prediction
input_data = {
    'age': 45, 'sex': 1, 'cp': 0, 'trestbps': 130,
    'chol': 250, 'fbs': 0, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 1.0, 'slope': 0, 'ca': 0, 'thal': 1,
    'height': 175, 'weight': 80
}

explanation = model.explain_prediction(input_data)
print(f"Risk Level: {explanation['risk_level']}")
print(f"Probability: {explanation['probability']:.1%}")
```

## 🏗️ Project Structure

```
Heart Disease Prediction T/
├── 📁 src/                          # Source code modules
│   ├── 📁 data/                     # Data-related modules
│   │   ├── feature_definitions.py   # Feature specifications & descriptions
│   │   └── __init__.py
│   ├── 📁 models/                   # Machine learning models
│   │   ├── heart_disease_model.py   # Main ML model implementation
│   │   ├── model.py                 # Alternative model interface
│   │   ├── feature_importance.py    # Feature analysis tools
│   │   └── __init__.py
│   ├── 📁 preprocessing/            # Data preprocessing
│   │   ├── preprocessor.py          # Data cleaning & transformation
│   │   └── __init__.py
│   ├── 📁 ui/                       # User interfaces
│   │   ├── gradio_interface.py      # Web interface implementation
│   │   ├── console_interface.py     # Command-line interface
│   │   └── __init__.py
│   ├── 📁 utils/                    # Utility functions
│   │   ├── data_loader.py           # Data loading & sample generation
│   │   └── __init__.py
│   └── __init__.py
├── 📁 models/                       # Trained model storage
│   └── heart_disease_model.joblib   # Saved model file
├── 📄 heart_disease_app.py          # Main web application
├── 📄 main.py                       # CLI application entry point
├── 📄 analyze_features.py           # Feature analysis script
├── 📄 build_exe.py                  # Executable builder
├── 📄 example.py                    # Usage examples
├── 📄 analyze_features_only.py      # Feature-only analysis script
├── 📄 simple_heart_app.py           # Simplified application version
├── 📄 requirements.txt              # Python dependencies
├── 📄 run_heart_disease_app.bat     # Windows launcher
├── 📄 build_exe.bat                 # Windows build script
├── 📄 feature_importance.png        # Generated feature importance plot
├── 📄 *.spec                        # PyInstaller specification files
└── 📄 README.md                     # This file
```

## 🔧 Dependencies

### Core Libraries
- **numpy==1.24.3** - Numerical computing
- **pandas==2.0.3** - Data manipulation
- **scikit-learn==1.3.0** - Machine learning algorithms
- **joblib==1.3.1** - Model serialization

### Visualization
- **matplotlib==3.7.2** - Plotting and visualization
- **seaborn==0.12.2** - Statistical visualizations

### User Interface
- **gradio==4.19.2** - Web interface framework

### Deployment
- **pyinstaller==6.5.0** - Executable creation

## 🎯 Key Algorithms & Techniques

### Machine Learning
- **Random Forest Classifier** - Ensemble learning for robust predictions
- **GridSearchCV** - Hyperparameter optimization
- **Cross-Validation** - Model validation and selection
- **Feature Importance** - Understanding model decisions

### Data Processing
- **One-Hot Encoding** - Categorical variable handling
- **Standard Scaling** - Numerical feature normalization
- **Missing Value Handling** - Data quality assurance
- **Feature Engineering** - BMI calculation and derived features

### Visualization Techniques
- **Risk Gauges** - Intuitive probability display
- **Feature Impact Charts** - Contribution analysis
- **Category Grouping** - Medical domain organization
- **Color Coding** - Risk level visualization

## 🎨 Visualization Features

### Enhanced Risk Dashboard
- **Professional Medical Theme** with clean white backgrounds
- **Interactive Risk Gauge** with color-coded severity (Green → Red)
- **Feature Impact Visualization** showing positive/negative contributions
- **Category-based Analysis** grouping features by medical domain
- **Personalized Recommendations** with medical icons and actionable advice
- **Real-time BMI Calculator** integrated into the interface
- **Medical Context Explanations** for each feature and risk factor

### Feature Categories
- 🎂 **Demographics** - Age, Gender, BMI
- 💓 **Symptoms** - Chest pain patterns
- 🩺 **Vital Signs** - Blood pressure, heart rate
- 🧪 **Blood Tests** - Cholesterol, blood sugar
- 📈 **Heart Tests** - ECG results
- 🏃 **Exercise Tests** - Stress test results
- 🔍 **Imaging** - Vessel blockages, perfusion
- 🏃‍♂️ **Physical** - Height, weight, derived metrics

### Recent Improvements
- ✅ **Fixed matplotlib title font conflicts** for better compatibility
- ✅ **Clean white backgrounds** throughout all visualizations
- ✅ **Enhanced medical styling** with professional appearance
- ✅ **Improved error handling** and user feedback

## 📊 Model Performance

The Random Forest model achieves:
- **High Accuracy** on validation datasets
- **Balanced Precision/Recall** for both classes
- **Robust Feature Importance** rankings
- **Reliable Probability Estimates** for risk assessment

*Note: Actual performance metrics depend on the training dataset used.*

## 🛡️ Medical Disclaimer

⚠️ **IMPORTANT**: This application is designed for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical feature definitions based on established cardiology research
- UCI Heart Disease Dataset for reference
- Scikit-learn community for machine learning tools
- Gradio team for the excellent web interface framework

## 🔧 Troubleshooting

### Common Issues

**Matplotlib Font Errors**
- Fixed: Title font conflicts resolved in latest version
- All backgrounds now use clean white styling

**Model Not Found**
- Run `python main.py` to train a new model automatically
- Model will be saved to `models/heart_disease_model.joblib`

**Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

**Executable Build Issues**
- Use `python build_exe.py` instead of direct PyInstaller commands
- Ensure all dependencies are properly installed

### Performance Tips
- Use the web interface for best user experience
- Console interface is faster for batch predictions
- Feature analysis scripts help understand model behavior

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `src/` modules
- Review the example usage in `example.py`
- Test with different interfaces to find what works best for your use case

### Available Interfaces Summary
1. **heart_disease_app.py** - Full-featured web interface (Recommended)
2. **main.py** - Command-line interface with training options
3. **analyze_features.py** - Feature analysis with web UI
4. **analyze_features_only.py** - Feature analysis without UI
5. **simple_heart_app.py** - Minimal prediction interface

---

**Made with ❤️ for better health outcomes** 
⭐ **Star this repository if you find it useful!** ⭐ 