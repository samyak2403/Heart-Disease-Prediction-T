# ❤️ Heart Disease Prediction System

## 🌟 Developed by Samyak Kamble

![Heart Disease Prediction](https://img.shields.io/badge/Health-AI-red?style=for-the-badge&logo=heart)

[<img src="https://static-00.iconduck.com/assets.00/visual-studio-code-icon-2048x2026-9ua8vqiy.png" alt="Get it on GitHub" height="80">](https://github.dev/samyak2403/Heart-Disease-Prediction-T)


A comprehensive machine learning system for predicting heart disease risk with an interactive and user-friendly interface.

## ✨ Key Features

- 🩺 **Advanced Risk Prediction**: Uses machine learning to predict heart disease risk with high accuracy
- 📊 **Interactive Visualizations**: Dynamic feature impact graphs showing how each factor affects risk
- 🎯 **Personalized Recommendations**: Custom health suggestions based on individual risk factors
- 📱 **User-Friendly Interface**: Large text and intuitive design for easy use by all age groups
- 🔍 **Real-time BMI Calculation**: Instantly calculates and categorizes BMI from height and weight
- 🌐 **Shareable Public Link**: Access the tool from anywhere via a public URL
- 💻 **Standalone Application**: Can be packaged as an executable (.exe) for offline use
- 📈 **Educational Information**: Provides medical context for each risk factor

## 📋 Medical Factors Analyzed

| Factor | Description | Impact |
|--------|-------------|--------|
| 👴 Age | Risk doubles every decade after 45 | High |
| ♂️ Gender | Men have 2-3x higher risk before age 55 | Medium |
| 💔 Chest Pain | Typical angina strongly indicates coronary artery disease | Very High |
| 🩸 Blood Pressure | Each 20mmHg increase doubles risk | High |
| 🍔 Cholesterol | 23% increased risk per 40mg/dl above 200 | High |
| 🧪 Blood Sugar | Diabetes doubles heart disease risk | Medium |
| 📈 ECG Results | ST-T abnormalities indicate 5x higher risk | High |
| ❤️ Max Heart Rate | Lower max heart rate indicates decreased function | Medium |
| 😣 Exercise Angina | Presence indicates 3x higher risk | High |
| 📉 ST Depression | Values >2mm indicate severe ischemia | High |
| 📊 ST Slope | Downsloping indicates poor prognosis | Medium |
| 🩺 Major Vessels | Risk increases 2x per vessel affected | High |
| 🔬 Thalassemia | Reversible defects indicate 3x higher risk | High |
| ⚖️ BMI | BMI >30 increases risk by 50% | Medium |

## 🧠 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Features**: 13 clinical parameters + BMI
- **Metrics**:
  - Accuracy: ~85%
  - Precision: ~84%
  - Recall: ~86%
  - F1 Score: ~85%
  - ROC AUC: ~90%

## 🖥️ Technical Implementation

### 📚 Project Structure
```
Heart Disease Prediction/
  ├── models/                  # Trained model files
  ├── src/                     # Source code
  │   ├── data/                # Data definitions and processing
  │   ├── models/              # ML model implementation
  │   ├── preprocessing/       # Data preprocessing
  │   ├── ui/                  # User interfaces
  │   └── utils/               # Utility functions
  ├── heart_disease_app.py     # Main application
  ├── build_exe.py             # Executable builder
  ├── main.py                  # CLI version
  └── requirements.txt         # Dependencies
```

### 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Data visualization
- **Gradio**: Web interface
- **PyInstaller**: Executable packaging

## 📦 Installation

### Requirements

```
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.1
gradio>=3.0.0
joblib>=1.0.1
pyinstaller>=5.0.0  # For executable creation
```

### 🚀 Quick Start

1. **Clone the repository**
   ```
   git clone https://github.com/samyak2403/Heart-Disease-Prediction-T.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```
   python heart_disease_app.py
   ```

4. **Build executable (optional)**
   ```
   python build_exe.py
   ```

## 💻 Usage

### Web Interface

1. Enter patient information in the form
2. Click "SUBMIT" to generate prediction
3. View results including:
   - Heart disease risk prediction
   - Risk level assessment
   - Contributing factors
   - Interactive feature impact visualization
   - Personalized health recommendations

### Executable Version

1. Run the generated .exe file
2. Follow the same steps as the web interface

## 📊 Feature Impact Visualization

The system provides a detailed visualization of how each factor contributes to heart disease risk:

- 📈 **Color-coded bars**: Red for risk-increasing factors, green for risk-decreasing factors
- 🔢 **Numerical impact**: Precise quantification of each factor's contribution
- ℹ️ **Educational tooltips**: Medical information about each risk factor
- 🎯 **Risk gauge**: Visual representation of overall heart disease risk
- 💡 **Personalized recommendations**: Tailored health advice based on risk factors

## 🔄 Model Training

The model is trained on a comprehensive dataset of heart disease cases with the following steps:

1. Data preprocessing and normalization
2. Feature engineering and selection
3. Model training with cross-validation
4. Hyperparameter optimization
5. Performance evaluation on test data

## 👨‍💻 Developer Information

This project was developed by **Samyak Kamble** as a comprehensive heart disease risk assessment tool combining medical knowledge with advanced machine learning techniques.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions, suggestions, or collaborations, please contact:
- **Samyak Kamble**
- Email: samyak.kamble@example.com
- LinkedIn: [linkedin.com/in/samyak-kamble](https://linkedin.com/in/samyakkamble)

---

⭐ **Star this repository if you find it useful!** ⭐ 
