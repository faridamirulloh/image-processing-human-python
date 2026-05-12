"""
Skrip build untuk membuat executable Windows
"""

import os
import sys
import subprocess
import shutil


def build():
    """Build executable Windows menggunakan PyInstaller"""
    
    # Project paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(project_dir, "src")
    models_dir = os.path.join(project_dir, "models")
    compat_dir = os.path.join(src_dir, "yolo_mp_compat")
    dist_dir = os.path.join(project_dir, "dist")
    build_dir = os.path.join(project_dir, "build")
    
    # Bersihkan build sebelumnya
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    # Periksa apakah PyInstaller tersedia
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: PyInstaller is not installed. Run 'pip install pyinstaller' first.")
        sys.exit(1)
    
    # Konstruksi perintah PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=FireDetectionApp",
        "--onefile",
        "--windowed",
        "--noconfirm",
        # Impor tersembunyi untuk ultralytics dan torch
        "--hidden-import=ultralytics",
        "--hidden-import=torch",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        # Kumpulkan file data ultralytics
        "--collect-data=ultralytics",
        "--collect-submodules=ultralytics",
        # Tambahkan direktori sumber ke path
        f"--paths={src_dir}",
    ]
    
    # Tambahkan model file jika ada
    model_file = "YOLO-MP.pt"
    model_path = os.path.join(models_dir, model_file)
    
    if os.path.exists(model_path):
        cmd.append(f"--add-data={model_path};models")
        print(f"Including model: {model_file}")
    else:
        print(f"WARNING: Model file {model_file} not found in models!")

    if os.path.exists(compat_dir):
        cmd.append(f"--add-data={compat_dir};yolo_mp_compat")
        print("Including YOLO-MP compatibility modules")
    else:
        print("WARNING: YOLO-MP compatibility modules not found in src/yolo_mp_compat!")
    
    # Titik masuk (harus yang terakhir)
    cmd.append(os.path.join(src_dir, "main.py"))
    
    # Tambahkan ikon jika ada
    icon_path = os.path.join(project_dir, "assets", "icon.ico")
    if os.path.exists(icon_path):
        cmd.insert(-1, f"--icon={icon_path}")
    
    print("Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    # Jalankan PyInstaller
    result = subprocess.run(cmd, cwd=project_dir)
    
    if result.returncode == 0:
        print("\n" + "="*50)
        print("BUILD SUCCESSFUL!")
        print("="*50)
        print(f"\nExecutable created at: {os.path.join(dist_dir, 'FireDetectionApp.exe')}")
        print("\nNote: The first run may take longer as it initializes the model.")
    else:
        print("\n" + "="*50)
        print("BUILD FAILED!")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    build()
