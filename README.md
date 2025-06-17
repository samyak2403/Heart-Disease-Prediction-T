# Heart Disease Prediction System

A Python-based system for predicting heart disease risk using machine learning.

[<img src="https://static-00.iconduck.com/assets.00/visual-studio-code-icon-2048x2026-9ua8vqiy.png" alt="Get it on GitHub" height="80">](https://github.dev/samyak2403/Heart-Disease-Prediction-T)


## Features

- Predicts heart disease risk based on patient data
- Provides risk level assessment and contributing factors
- Uses Random Forest classifier for predictions
- Includes data preprocessing and validation
- Console interface for easy interaction
- Feature importance analysis to identify key predictors
- Interactive Gradio web UI for predictions
- Executable version with direct web UI access

## Installation

```
pip install -r requirements.txt
```

## Usage

### Console Interface

Run the main script:

```
python main.py
```

This will train a model (or use existing one), run a test prediction, and offer to launch the interactive console interface.

### Feature Importance Analysis

To analyze which features are most important for prediction:

```
python analyze_features.py
```

This will:
1. Train a model (or use existing one)
2. Analyze and display feature importance
3. Show a visualization of feature importance
4. Launch the Gradio UI with the most important features

### Gradio Web Interface

To launch just the web interface:

```
python run_gradio_ui.py
```

This provides a user-friendly web interface for making predictions, showing:
- Risk prediction result
- Risk level assessment
- Contributing factors
- Feature impact visualization

### Executable Version

For users who don't have Python installed, you can create a standalone executable:

1. Run the build script:
   ```
   python build_exe.py
   ```
   or simply double-click `build_exe.bat`

2. The executable will be created in the `dist` folder
3. Distribute the entire `dist` folder to users
4. Users can run the application by double-clicking `HeartDiseasePrediction.exe`

The executable version:
- Opens a web interface directly in the user's browser
- Provides the same functionality as the Gradio web interface
- Requires no Python installation or technical knowledge
- Shows feature importance visualization for predictions

## Disclaimer

For educational purposes only. Not for actual medical diagnosis. 
