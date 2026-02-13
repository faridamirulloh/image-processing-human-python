"""
Build script for creating Windows executable
"""

import os
import sys
import subprocess
import shutil


def build():
    """Build the Windows executable using PyInstaller"""
    
    # Project paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_dir, "src")
    dist_dir = os.path.join(project_dir, "dist")
    build_dir = os.path.join(project_dir, "build")
    
    # Clean previous builds
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    # Check if PyInstaller is available
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: PyInstaller is not installed. Run 'pip install pyinstaller' first.")
        sys.exit(1)
    
    # PyInstaller command construction
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=HumanDetectionApp",
        "--onefile",
        "--windowed",
        "--noconfirm",
        # Hidden imports for ultralytics and torch
        "--hidden-import=ultralytics",
        "--hidden-import=torch",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        # Collect ultralytics data files
        "--collect-data=ultralytics",
        # Add source directory to path
        f"--paths={src_dir}",
    ]
    
    # Add YOLO model files if they exist (only nano and small for size)
    model_files = [
        "yolov8n.pt", "yolov8s.pt",
        "yolo11n.pt", "yolo11s.pt", 
        "yolo12n.pt", "yolo12s.pt"
    ]
    
    included_models = []
    for model_file in model_files:
        model_path = os.path.join(project_dir, model_file)
        if os.path.exists(model_path):
            # Add model to bundled data (source;destination)
            cmd.append(f"--add-data={model_path};.")
            included_models.append(model_file)
            print(f"Including model: {model_file}")
    
    # Entry point (must be last)
    cmd.append(os.path.join(src_dir, "main.py"))
    
    # Add icon if it exists
    icon_path = os.path.join(project_dir, "assets", "icon.ico")
    if os.path.exists(icon_path):
        cmd.insert(-1, f"--icon={icon_path}")
    
    print("Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=project_dir)
    
    if result.returncode == 0:
        print("\n" + "="*50)
        print("BUILD SUCCESSFUL!")
        print("="*50)
        print(f"\nExecutable created at: {os.path.join(dist_dir, 'HumanDetectionApp.exe')}")
        print("\nNote: The first run may take longer as it downloads the YOLO model.")
    else:
        print("\n" + "="*50)
        print("BUILD FAILED!")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    build()
