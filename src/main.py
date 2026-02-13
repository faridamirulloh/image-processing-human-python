"""
Human Detection Application - Entry Point
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def main():
    """Main entry point for the application"""
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Human Detection App")
    app.setOrganizationName("HumanDetection")
    
    # Import here to avoid circular imports
    from app import MainWindow
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
