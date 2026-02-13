# Application Codebase & Flow Documentation

This document provides a comprehensive overview of the Human Detection Application, explaining the execution flow and detailing the purpose of each code file.

## 1. Application Flow Overview

1.  **Entry Point (`run.py` / `src/main.py`)**:
    *   The user runs `run.py`.
    *   It checks for the correct Python version (3.11).
    *   It sets up the Python path to include the `src` directory.
    *   It calls `main()` from `src/main.py`.
    *   `src/main.py` initializes the `QApplication` (PyQt framework) and launches the `MainWindow`.

2.  **Initialization (`src/app.py` - `MainWindow`)**:
    *   The `MainWindow` initializes the UI layout (Video Widget, Stats Widget, Control Bar).
    *   It instantiates core services:
        *   **CameraService**: Scans for connected cameras.
        *   **VideoService**: Prepares for background thread video capture.
        *   **RecordingService**: Sets up output paths for videos/screenshots.
    *   It triggers an async pre-loading of the **DetectorService** (YOLO model).

3.  **Runtime Loop**:
    *   **User Action**: User selects a camera and clicks "Start".
    *   **Video Capture**: `VideoService` runs in a separate thread, reading frames from the camera and emitting `frame_ready` signals.
    *   **Frame Processing (`app.py` -> `_on_frame_ready`)**:
        *   The main thread receives the frame.
        *   If **Detection** is active: The frame is passed to `DetectorService.detect_humans()`.
            *   YOLO inference runs (CPU).
            *   Bounding boxes are drawn.
            *   Person count is returned.
        *   If **Recording** is active: The frame is passed to `RecordingService.write_frame()`.
    *   **Display**: The processed frame (with annotations) is sent to `VideoWidget` for rendering.
    *   **Stats Update**: `StatsWidget` is updated with the latest FPS and person count.

## 2. Code Breakdown by Directory

### Root Directory

#### `run.py`
*   **What**: The developer entry point script.
*   **Why**: Ensures the environment is correct (Python 3.11) and paths are set up before importing the main app packages. Prevents "ModuleNotFound" errors.

#### `build.py`
*   **What**: Automation script for PyInstaller.
*   **Why**: packages the Python application into a standalone `.exe` file for Windows. It handles including hidden imports (like `ultralytics`, `torch`) and bundling assets (icons, model files).

### `src/` Directory

#### `src/main.py`
*   **What**: The formal entry point for the PyQt application.
*   **Why**: Separates the GUI startup logic (creating `QApplication`, setting organization names) from the window logic.

#### `src/app.py`
*   **What**: The Controller / Main Window.
*   **Why**: This is the "brain" of the UI. It orchestrates everything:
    *   Connects the UI buttons to functions (Start, Stop, Record).
    *   Receives signals from services (new video frames).
    *   Decides what to do with a frame (detect, record, display).

### `src/services/` Directory
*Logic that handles heavy lifting, separated from the UI.*

#### `camera_service.py`
*   **What**: Camera discovery utility.
*   **Why**: finding available cameras on Windows can be tricky. This service tries multiple backends (`DirectShow`, `MSMF`) to robustly list available webcams and their resolutions.

#### `video_service.py`
*   **What**: Background thread for video capture.
*   **Why**: Reading from a camera is a blocking operation. If done in the main UI thread, the application would freeze/lag. Using a `QThread` ensures the UI remains responsive while video is captured in parallel.

#### `detector_service.py`
*   **What**: Verify wrapper around the YOLO (Ultralytics) model.
*   **Why**: Encapsulates all AI logic.
    *   Handles loading the model.
    *   Runs inference (`model(frame)`).
    *   Filters results (keeps only "person" class).
    *   Implements **tracking & smoothing** (IoU logic) to prevent bounding boxes from flickering jitterily.

#### `recording_service.py`
*   **What**: Media I/O handler.
*   **Why**: Manages saving video files (`.mp4`) and screenshots (`.png`). Handles file naming (timestamps) and output directory management.

### `src/widgets/` Directory
*Reusable UI components.*

#### `video_widget.py`
*   **What**: The screen that displays the camera feed.
*   **Why**: A standard `QLabel` isn't enough. This widget handles:
    *   Aspect ratio preservation (so the video doesn't stretch).
    *   Efficient conversion from OpenCV (BGR) frames to Qt (RGB) images.
    *   Visual states (Loading, Error, No Camera).

#### `stats_widget.py`
*   **What**: The sidebar showing FPS, count, etc.
*   **Why**: Keeps the main `app.py` layout clean by componentizing the statistics display. Updating one stat (`update_fps`) acts on this widget.

### `src/utils/` Directory
*Helpers and configuration.*

#### `constants.py`
*   **What**: Global configuration variables.
*   **Why**: Avoids "magic numbers" in the code. Stores window size, default model selection, color definitions, and paths in one place for easy editing.

#### `styles.py`
*   **What**: CSS-like stylesheets (QSS) for PyQt.
*   **Why**: Keeps the UI styling (colors, fonts, borders) separate from the Python logic. Allows for easy theming (Dark Mode).
