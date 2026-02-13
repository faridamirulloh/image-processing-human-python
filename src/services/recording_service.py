"""
Recording Service - Handles video recording and screenshot capture.
Saves output files to a user-configurable output folder.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from utils.constants import DEFAULT_OUTPUT_FOLDER, RECORDING_FPS, RECORDING_CODEC


class RecordingService:
    """
    Service for recording video and capturing screenshots.
    Output files are saved to a configurable folder with timestamped filenames.
    """

    def __init__(self, output_folder: str = DEFAULT_OUTPUT_FOLDER):
        self._output_folder = output_folder
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording = False
        self._current_file = ""

        # Ensure output folder exists
        os.makedirs(self._output_folder, exist_ok=True)

    # -------------------------------------------------------------------------
    # Output Folder Management
    # -------------------------------------------------------------------------

    def set_output_folder(self, path: str):
        """Change the output folder and create it if needed."""
        self._output_folder = path
        os.makedirs(self._output_folder, exist_ok=True)

    def get_output_folder(self) -> str:
        """Return the current output folder path."""
        return self._output_folder

    def open_output_folder(self):
        """Open the output folder in the system file explorer."""
        os.makedirs(self._output_folder, exist_ok=True)
        os.startfile(self._output_folder)

    # -------------------------------------------------------------------------
    # Video Recording
    # -------------------------------------------------------------------------

    def start_recording(self, width: int, height: int) -> str:
        """
        Start recording video to an .mp4 file.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Path to the output file being written
        """
        if self._is_recording:
            return self._current_file

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(self._output_folder, filename)

        # Create VideoWriter with configured codec and FPS
        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        self._writer = cv2.VideoWriter(filepath, fourcc, RECORDING_FPS, (width, height))

        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Failed to create video writer for: {filepath}")

        self._is_recording = True
        self._current_file = filepath
        return filepath

    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to the active recording.

        Args:
            frame: BGR frame from OpenCV (same format as camera output)
        """
        if self._is_recording and self._writer is not None:
            self._writer.write(frame)

    def stop_recording(self) -> str:
        """
        Stop the active recording and release the writer.

        Returns:
            Path to the saved recording file
        """
        saved_file = self._current_file

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._is_recording = False
        self._current_file = ""
        return saved_file

    def is_recording(self) -> bool:
        """Check if a recording is currently active."""
        return self._is_recording

    # -------------------------------------------------------------------------
    # Screenshot Capture
    # -------------------------------------------------------------------------

    def capture_screenshot(self, frame: np.ndarray) -> str:
        """
        Save the current frame as a PNG screenshot.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Path to the saved screenshot file
        """
        if frame is None:
            raise ValueError("No frame available to capture")

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        filepath = os.path.join(self._output_folder, filename)

        cv2.imwrite(filepath, frame)
        return filepath

    def cleanup(self):
        """Release any active resources."""
        if self._is_recording:
            self.stop_recording()
