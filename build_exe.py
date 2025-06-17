"""
Build script to create an executable for the Heart Disease Prediction System.
"""

import os
import sys
import subprocess
import shutil

def build_executable():
    """Build the executable using PyInstaller."""
    print("Building executable for Heart Disease Prediction System...")
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Find PyInstaller executable
    pyinstaller_path = os.path.join(os.path.dirname(sys.executable), "Scripts", "pyinstaller.exe")
    if not os.path.exists(pyinstaller_path):
        # Try alternative location for Windows Store Python
        alt_path = os.path.join(
            os.path.dirname(os.path.dirname(sys.executable)), 
            "LocalCache", "local-packages", 
            f"Python{sys.version_info.major}{sys.version_info.minor}", 
            "Scripts", "pyinstaller.exe"
        )
        if os.path.exists(alt_path):
            pyinstaller_path = alt_path
        else:
            # Last resort: use module directly
            pyinstaller_path = [sys.executable, "-m", "PyInstaller"]
    
    print(f"Using PyInstaller at: {pyinstaller_path}")
    
    # Clean up previous builds
    if os.path.exists("dist"):
        print("Cleaning up previous builds...")
        shutil.rmtree("dist", ignore_errors=True)
    if os.path.exists("build"):
        shutil.rmtree("build", ignore_errors=True)
    
    # Create the spec file
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['heart_disease_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sklearn.ensemble._forest',
        'sklearn.tree._tree',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'matplotlib.backends.backend_tkagg',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.skiplist',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add data files
a.datas += [
    ('models/heart_disease_model.joblib', 'models/heart_disease_model.joblib', 'DATA'),
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HeartDiseasePrediction',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='heart_icon.ico' if os.path.exists('heart_icon.ico') else None,
)
"""
    
    with open("heart_disease_app.spec", "w") as f:
        f.write(spec_content)
    
    # Build the executable
    print("Running PyInstaller...")
    
    if isinstance(pyinstaller_path, list):
        # Using Python module
        cmd = pyinstaller_path + ["--clean", "--noconfirm", "heart_disease_app.spec"]
    else:
        # Using executable
        cmd = [pyinstaller_path, "--clean", "--noconfirm", "heart_disease_app.spec"]
    
    try:
        subprocess.check_call(cmd)
        print("\nBuild completed!")
        print("Executable can be found in the 'dist' folder.")
        
        # Add a README file to the dist folder
        readme_content = """
Heart Disease Prediction System
==============================

This application helps predict the risk of heart disease based on patient data.

How to Use:
1. Double-click the HeartDiseasePrediction.exe file to start the application
2. Choose one of the available options:
   - Launch Web Interface: Opens a web-based interface in your browser
   - Analyze Feature Importance: Shows which factors contribute most to heart disease risk
   - Launch Console Interface: Opens a command-line interface for prediction

Note: This application is for educational purposes only and should not be used for actual medical diagnosis.
"""
        
        os.makedirs("dist", exist_ok=True)
        with open(os.path.join("dist", "README.txt"), "w") as f:
            f.write(readme_content)
        
        print("Added README.txt to the dist folder.")
        
        # Check if the build was successful
        exe_path = os.path.join("dist", "HeartDiseasePrediction.exe")
        if os.path.exists(exe_path):
            print(f"\nSuccess! Executable created at: {os.path.abspath(exe_path)}")
            print("You can distribute the entire 'dist' folder to users.")
        else:
            print("\nError: Executable was not created successfully.")
            print("Check the build output for errors.")
            
    except subprocess.CalledProcessError as e:
        print(f"\nError running PyInstaller: {e}")
        print("Try running the following command manually:")
        if isinstance(pyinstaller_path, list):
            print(f"{' '.join(pyinstaller_path)} --clean --noconfirm heart_disease_app.spec")
        else:
            print(f"{pyinstaller_path} --clean --noconfirm heart_disease_app.spec")

if __name__ == "__main__":
    build_executable() 