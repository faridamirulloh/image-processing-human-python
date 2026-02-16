"""
Widget Statistik - Menampilkan statistik deteksi real-time.
Menampilkan jumlah orang, FPS, info model, dan status deteksi.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class StatsWidget(QWidget):
    """
    Widget untuk menampilkan statistik deteksi.
    Dirancang agar ringkas dan responsif untuk sidebar kanan.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """Inisialisasi komponen UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Judul
        title = QLabel("📊 Statistics")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #00d9ff;")
        layout.addWidget(title)
        
        # Garis pemisah
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #2d2d44;")
        layout.addWidget(separator)
        
        # Kartu statistik
        self._person_count_label = self._create_stat_card("👥 Persons", "0", "#00ff88")
        layout.addWidget(self._person_count_label)
        
        self._fps_label = self._create_stat_card("📈 FPS", "0.0", "#ffbb00")
        layout.addWidget(self._fps_label)
        
        self._model_label = self._create_stat_card("🧠 Model", "YOLOv8n", "#a55eea")
        layout.addWidget(self._model_label)
        
        self._status_label = self._create_stat_card("📍 Status", "Stopped", "#8b8b8b")
        layout.addWidget(self._status_label)
        
        layout.addStretch()  # Dorong kartu ke atas
        
        # Gaya widget
        self.setStyleSheet("""
            StatsWidget {
                background-color: #16213e;
                border-radius: 10px;
            }
        """)
        self.setMinimumWidth(120)
    
    def _create_stat_card(self, title: str, value: str, color: str) -> QFrame:
        """
        Buat kartu statistik dengan judul dan tampilan nilai berwarna.
        
        Args:
            title: Judul kartu
            value: Nilai awal untuk ditampilkan
            color: Warna untuk teks nilai
            
        Returns:
            QFrame berisi tata letak kartu statistik
        """
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        
        # Label judul (lebih kecil, abu-abu)
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 9))
        title_label.setStyleSheet("color: #8b8b8b;")
        layout.addWidget(title_label)
        
        # Label nilai (lebih besar, berwarna)
        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        value_label.setStyleSheet(f"color: {color};")
        value_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(value_label)
        
        # Simpan referensi untuk pembaruan nanti
        card.value_label = value_label
        card.color = color
        
        return card
    
    def update_person_count(self, count: int):
        """Perbarui tampilan jumlah orang"""
        self._person_count_label.value_label.setText(str(count))
    
    def update_fps(self, fps: float):
        """Perbarui tampilan FPS"""
        self._fps_label.value_label.setText(f"{fps:.1f}")
    
    def update_model(self, model: str):
        """Perbarui tampilan model (menampilkan nama model yang dipersingkat)"""
        short_name = model.split(" - ")[0] if " - " in model else model
        self._model_label.value_label.setText(short_name)
    
    def update_status(self, status: str, is_active: bool = False):
        """Perbarui tampilan status dengan warna yang sesuai"""
        self._status_label.value_label.setText(status)
        color = "#00ff88" if is_active else "#8b8b8b"
        self._status_label.value_label.setStyleSheet(f"color: {color};")
    
    def reset_stats(self):
        """Atur ulang semua statistik ke nilai default"""
        self.update_person_count(0)
        self.update_fps(0.0)
        self.update_status("Stopped", False)
