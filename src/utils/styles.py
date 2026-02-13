"""
UI Styles and Themes
Centralized stylesheet definitions for the application.
"""

def get_main_theme() -> str:
    """Get the global application theme stylesheet."""
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
    """Get stylesheet for combo boxes."""
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
    """Generate stylesheet for standard action buttons."""
    return f"""
        QPushButton {{
            background-color: {color};
            border: none;
            border-radius: 6px;
            color: #000000;
            font-weight: bold;
            font-size: 14px;
            padding: 8px 16px;
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
    """Generate stylesheet for icon-only buttons."""
    return f"""
        QPushButton {{
            background-color: {color};
            border: none;
            border-radius: 6px;
            color: #ffffff;
            font-weight: bold;
            font-size: 18px;
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
    """Get stylesheet for QSplitter handle."""
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
    """Get stylesheet for the top control bar frame."""
    return """
        QFrame {
            background-color: #16213e;
            border-radius: 8px;
        }
    """

def get_status_bar_style() -> str:
    """Get stylesheet for the status bar."""
    return "color: #8b8b8b; font-size: 11px;"
