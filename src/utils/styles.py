"""
Gaya dan Tema UI
Definisi stylesheet terpusat untuk aplikasi.
"""

def get_main_theme() -> str:
    """Dapatkan stylesheet tema aplikasi global."""
    return """
        QMainWindow {
            background-color: #0f0f1a;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QToolTip {
            background-color: #1a1a2e;
            color: #ffffff;
            border: 1px solid #2d2d44;
            border-radius: 4px;
            padding: 5px;
        }
    """

def get_combo_style() -> str:
    """Dapatkan stylesheet untuk kotak kombo."""
    return """
        QComboBox {
            background-color: #1a1a2e;
            border: 2px solid #2d2d44;
            border-radius: 6px;
            padding: 8px 12px;
            color: #ffffff;
            font-size: 13px;
        }
        QComboBox:hover {
            border-color: #00d9ff;
        }
        QComboBox::drop-down {
            border: none;
            padding-right: 10px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #00d9ff;
            margin-right: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #1a1a2e;
            border: 2px solid #2d2d44;
            color: #ffffff;
            selection-background-color: #00d9ff;
            selection-color: #000000;
        }
    """

def get_button_style(color: str, hover_color: str) -> str:
    """Hasilkan stylesheet untuk tombol tindakan standar."""
    return f"""
        QPushButton {{
            background-color: {color};
            border: none;
            border-radius: 6px;
            color: #000000;
            font-weight: bold;
            font-size: 14px;
            padding: 4px 8px 4px 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        QPushButton:hover {{
            background-color: {hover_color};
        }}
        QPushButton:pressed {{
            background-color: {color};
        }}
        QPushButton:disabled {{
            background-color: #2d2d44;
            color: #5a5a5a;
        }}
    """

def get_icon_button_style(color: str, hover_color: str) -> str:
    """Hasilkan stylesheet untuk tombol hanya icon."""
    return f"""
        QPushButton {{
            background-color: {color};
            border: none;
            border-radius: 6px;
            color: #ffffff;
            font-weight: bold;
            font-size: 18px;
            padding: 4px 8px 8px 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        QPushButton:hover {{
            background-color: {hover_color};
        }}
        QPushButton:disabled {{
            background-color: #2d2d44;
            color: #5a5a5a;
        }}
    """

def get_splitter_handle_style() -> str:
    """Dapatkan stylesheet untuk pegangan QSplitter."""
    return """
        QSplitter::handle {
            background-color: #2d2d44;
            border-radius: 2px;
        }
        QSplitter::handle:hover {
            background-color: #00d9ff;
        }
    """

def get_control_bar_style() -> str:
    """Dapatkan stylesheet untuk bingkai bilah kontrol atas."""
    return """
        QFrame {
            background-color: #16213e;
            border-radius: 8px;
        }
    """

def get_status_bar_style() -> str:
    """Dapatkan stylesheet untuk bilah status."""
    return "color: #8b8b8b; font-size: 11px;"

def get_settings_dialog_style() -> str:
    """Get stylesheet for the settings dialog."""
    return """
        QDialog {
            background-color: #0f0f1a;
        }
        QLabel {
            color: #ffffff;
            font-size: 13px;
        }
        QGroupBox {
            color: #00d9ff;
            font-size: 13px;
            font-weight: bold;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
        }
        QSpinBox {
            background-color: #1a1a2e;
            border: 2px solid #2d2d44;
            border-radius: 6px;
            padding: 6px 10px;
            color: #ffffff;
            font-size: 13px;
            min-width: 60px;
        }
        QSpinBox:hover {
            border-color: #00d9ff;
        }
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: #2d2d44;
            border: none;
            width: 20px;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background-color: #00d9ff;
        }
        QCheckBox {
            color: #ffffff;
            font-size: 13px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid #2d2d44;
            background-color: #1a1a2e;
        }
        QCheckBox::indicator:checked {
            background-color: #00d9ff;
            border-color: #00d9ff;
        }
        QCheckBox::indicator:hover {
            border-color: #00d9ff;
        }
    """

def get_slider_style() -> str:
    """Get stylesheet for QSlider controls."""
    return """
        QSlider::groove:horizontal {
            background: #2d2d44;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #00d9ff;
            border: none;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #00b8d9;
        }
        QSlider::sub-page:horizontal {
            background: #00d9ff;
            border-radius: 3px;
        }
    """
