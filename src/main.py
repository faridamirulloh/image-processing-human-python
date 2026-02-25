"""
Aplikasi Deteksi Manusia - Entry Point
Menangani inisialisasi aplikasi, optimisasi CPU, dan error startup.
"""

import sys
import os
import traceback

# Periksa dependensi kritis sebelum inisialisasi Qt
try:
    import torch
except ImportError:
    print(
        "WARNING: PyTorch tidak terinstall. "
        "Deteksi AI tidak akan berfungsi.\n"
        "Install dengan: pip install torch torchvision"
    )
except Exception as e:
    print(f"WARNING: Gagal memuat PyTorch: {e}")

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt


def _boost_process_priority():
    """
    Optimalkan prioritas proses dan threading OpenCV secara dinamis.
    Menyesuaikan agresivitas berdasarkan jumlah core CPU perangkat:
    - 8+ core: HIGH priority, semua core untuk OpenCV
    - 4-7 core: ABOVE_NORMAL priority, sisakan 1 core
    - 1-3 core: NORMAL priority, gunakan semua yang ada
    """
    import cv2
    
    cpu_count = os.cpu_count() or 2
    
    # Atur thread OpenCV secara dinamis
    if cpu_count >= 8:
        cv2.setNumThreads(cpu_count)
    elif cpu_count >= 4:
        cv2.setNumThreads(cpu_count - 1)  # Sisakan 1 untuk UI
    else:
        cv2.setNumThreads(cpu_count)  # Gunakan semua yang ada
    
    # Tingkatkan prioritas proses di Windows
    if sys.platform == 'win32':
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            
            if cpu_count >= 8:
                # HIGH_PRIORITY_CLASS = 0x0080
                priority = 0x0080
                priority_name = "HIGH"
            elif cpu_count >= 4:
                # ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
                priority = 0x8000
                priority_name = "ABOVE_NORMAL"
            else:
                # NORMAL — jangan agresif di perangkat lemah
                priority = 0x0020
                priority_name = "NORMAL"
            
            ctypes.windll.kernel32.SetPriorityClass(handle, priority)
            print(f"Process priority: {priority_name} ({cpu_count} CPU cores, OpenCV threads: {cv2.getNumThreads()})")
        except Exception as e:
            print(f"Could not set process priority: {e}")
    else:
        try:
            nice_val = -10 if cpu_count >= 8 else (-5 if cpu_count >= 4 else 0)
            if nice_val < 0:
                os.nice(nice_val)
            print(f"Process priority boosted ({cpu_count} CPU cores)")
        except Exception:
            pass


def main():
    """Titik masuk utama untuk aplikasi"""
    # Aktifkan penskalaan DPI tinggi
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Buat aplikasi
    app = QApplication(sys.argv)
    app.setApplicationName("Human Detection App")
    app.setOrganizationName("HumanDetection")
    
    # Optimalkan CPU: prioritas tinggi + semua core
    _boost_process_priority()
    
    try:
        # Impor di sini untuk menghindari impor circular
        from app import MainWindow
        
        # Buat dan tampilkan jendela utama
        window = MainWindow()
        window.show()
        
    except Exception as e:
        # Tampilkan error fatal sebagai dialog agar terlihat oleh pengguna
        error_msg = (
            f"Gagal memulai aplikasi:\n\n"
            f"{type(e).__name__}: {e}\n\n"
            f"Detail teknis:\n{traceback.format_exc()}"
        )
        print(error_msg)
        
        QMessageBox.critical(
            None,
            "Error - Human Detection App",
            f"Aplikasi gagal dimulai:\n\n{e}\n\n"
            "Pastikan semua dependensi terinstall:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)
    
    # Jalankan loop event
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

