"""
Layanan Perekaman - Menangani perekaman video dan tangkapan layar.
Menyimpan file output ke folder output yang dapat dikonfigurasi pengguna.
"""

import os
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from utils.constants import DEFAULT_OUTPUT_FOLDER, RECORDING_FPS, RECORDING_CODEC


class RecordingService:
    """
    Layanan untuk merekam video dan mengambil tangkapan layar.
    File output disimpan ke folder yang dapat dikonfigurasi dengan nama file bertanggal.
    """

    def __init__(self, output_folder: str = DEFAULT_OUTPUT_FOLDER):
        self._output_folder = output_folder
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording = False
        self._current_file = ""
        self._rec_start_time = 0.0
        self._frames_written = 0
        self._last_frame = None

        # Pastikan folder output ada
        os.makedirs(self._output_folder, exist_ok=True)

    # -------------------------------------------------------------------------
    # Manajemen Folder Output
    # -------------------------------------------------------------------------

    def set_output_folder(self, path: str):
        """Ubah folder output dan buat jika diperlukan."""
        self._output_folder = path
        os.makedirs(self._output_folder, exist_ok=True)

    def get_output_folder(self) -> str:
        """Kembalikan path folder output saat ini."""
        return self._output_folder

    def open_output_folder(self):
        """Buka folder output di file explorer sistem."""
        os.makedirs(self._output_folder, exist_ok=True)
        os.startfile(self._output_folder)

    # -------------------------------------------------------------------------
    # Perekaman Video
    # -------------------------------------------------------------------------

    def start_recording(self, width: int, height: int) -> str:
        """
        Mulai merekam video ke file .mp4.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Path to the output file being written
        """
        if self._is_recording:
            return self._current_file

        # Hasilkan nama file bertanggal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(self._output_folder, filename)

        # Buat VideoWriter dengan codec dan FPS yang dikonfigurasi
        fourcc = cv2.VideoWriter_fourcc(*RECORDING_CODEC)
        self._writer = cv2.VideoWriter(filepath, fourcc, RECORDING_FPS, (width, height))

        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"Failed to create video writer for: {filepath}")

        self._is_recording = True
        self._current_file = filepath
        self._rec_start_time = time.time()
        self._frames_written = 0
        self._last_frame = None
        return filepath

    def write_frame(self, frame: np.ndarray):
        """
        Tulis satu frame ke perekaman aktif.
        Menggunakan wall-clock timestamp untuk menjaga sinkronisasi waktu.
        Frame yang terlewat diisi dengan duplikat frame sebelumnya.

        Args:
            frame: BGR frame from OpenCV (same format as camera output)
        """
        if not self._is_recording or self._writer is None:
            return

        now = time.time()
        elapsed = now - self._rec_start_time
        expected_frame = int(elapsed * RECORDING_FPS)

        # Don't write if ahead of schedule
        if expected_frame <= self._frames_written:
            self._last_frame = frame
            return

        # Fill gaps with previous frame (cap at 2s to avoid huge burst)
        gap = min(expected_frame - self._frames_written - 1, int(RECORDING_FPS * 2))
        if gap > 0 and self._last_frame is not None:
            for _ in range(gap):
                self._writer.write(self._last_frame)
                self._frames_written += 1

        # Write current frame
        self._writer.write(frame)
        self._frames_written += 1
        self._last_frame = frame

    def stop_recording(self) -> str:
        """
        Hentikan perekaman aktif dan lepaskan perekam.

        Returns:
            Path to the saved recording file
        """
        saved_file = self._current_file

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._is_recording = False
        self._current_file = ""
        self._last_frame = None
        return saved_file

    def is_recording(self) -> bool:
        """Periksa apakah perekaman sedang aktif."""
        return self._is_recording

    # -------------------------------------------------------------------------
    # Tangkapan Layar
    # -------------------------------------------------------------------------

    def capture_screenshot(self, frame: np.ndarray) -> str:
        """
        Simpan frame saat ini sebagai screenshot PNG.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Path to the saved screenshot file
        """
        if frame is None:
            raise ValueError("No frame available to capture")

        # Hasilkan nama file bertanggal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        filepath = os.path.join(self._output_folder, filename)

        success = cv2.imwrite(filepath, frame)
        if not success:
            raise RuntimeError(f"Gagal menyimpan screenshot ke: {filepath}")
        return filepath

    def cleanup(self):
        """Lepaskan sumber daya aktif apa pun."""
        if self._is_recording:
            self.stop_recording()
