"""
Aplikasi Deteksi Manusia - Entry Point
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def main():
    """Titik masuk utama untuk aplikasi"""
    # Aktifkan penskalaan DPI tinggi
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Buat aplikasi
    app = QApplication(sys.argv)
    app.setApplicationName("Human Detection App")
    app.setOrganizationName("HumanDetection")
    
    # Impor di sini untuk menghindari impor circular
    from app import MainWindow
    
    # Buat dan tampilkan jendela utama
    window = MainWindow()
    window.show()
    
    # Jalankan loop event
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
