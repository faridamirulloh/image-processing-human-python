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
