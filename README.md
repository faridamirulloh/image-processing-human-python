# 📷 Human Detection & Counting App

A high-performance Windows desktop application for real-time human detection and counting using YOLO models (v8, v11, v12).

![App Icon](src/assets/icon.png)

## ✨ Features

- **Real-time Detection**: Detects humans with high accuracy using state-of-the-art YOLO models.
- **Multiple Camera Support**: Automatically scans and lists all connected webcams.
- **Recording & Snapshots**: Record video clips (`.mp4`) or take screenshots (`.png`) with one click.
- **Live Statistics**: Displays real-time FPS, person count, and active model info.
- **Optimized for CPU**: Designed to run efficiently on standard CPUs without requiring a dedicated GPU.

## 🚀 Quick Start

### Prerequisites
- **Windows 10 or 11** (64-bit)
- **Python 3.11** (Strictly required)

### Installation

1.  **Clone or Download** this repository.
2.  **Run the automated setup script**:
    ```powershell
    python run.py
    ```
    *Note: The script will check your environment and tell you exactly what to do if dependencies are missing.*

### Manual Setup (if needed)

If you prefer setting up manually:

1.  **Create Virtual Environment**:
    ```powershell
    py -3.11 -m venv venv311
    .\venv311\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```powershell
    python run.py
    ```

## 🖥️ Usage Guide

1.  **Select Camera**: Use the dropdown list to choose your input camera.
2.  **Select Model**:
    *   **Nano (n)**: Fastest, recommended for most PCs.
    *   **Small (s)**: More accurate, but requires a powerful CPU.
3.  **Controls**:
    *   **Start**: Begins AI detection.
    *   **Stop**: Pauses detection (camera preview remains active).
    *   **Record**: Toggles video recording to the output folder.
    *   **Folder Icon**: Opens the directory where recordings are saved.

## 🛠️ Building the Executable

To create a standalone `.exe` file for easy distribution:

```powershell
.\venv311\Scripts\activate
python build.py
```

The output file `HumanDetectionApp.exe` will appear in the `dist` folder.

## 📁 Project Structure

For a detailed explanation of the codebase and how it works, please see [CODE_OVERVIEW.md](CODE_OVERVIEW.md).

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| **"No module named..."** | Ensure you activated the virtual environment (`.\venv311\Scripts\activate`). |
| **Camera not found** | Click the **"R"** (Refresh) button. Check if another app (Zoom/Teams) is using the camera. |
| **Low FPS / Lag** | Switch to a **Nano** model (e.g., `YOLOv8n`). Ensure your laptop is plugged into power. |
| **Model download fails** | Check your internet connection. The app downloads models automatically on first use. |

## ⚖️ License
MIT License
