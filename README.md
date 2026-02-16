# 📷 Aplikasi Deteksi & Penghitungan Manusia

Aplikasi desktop Windows berkinerja tinggi untuk deteksi dan penghitungan manusia secara real-time menggunakan model YOLO (v8, v11, v12).

![Ikon Aplikasi](src/assets/icon.png)

## ✨ Fitur

- **Deteksi Real-time**: Mendeteksi manusia dengan akurasi tinggi menggunakan model YOLO.
- **Dukungan Banyak Kamera**: Memindai dan mencantumkan semua webcam yang terhubung secara otomatis.
- **Perekaman & Snapshot**: Rekam klip video (`.mp4`) atau ambil tangkapan layar (`.png`) dengan satu klik.
- **Statistik Langsung**: Menampilkan FPS, jumlah orang, dan info model aktif secara real-time.
- **Dioptimalkan untuk CPU**: Dirancang untuk berjalan secara efisien pada CPU standar tanpa memerlukan GPU khusus.

## 🚀 Mulai Cepat

### Prasyarat
- **Windows 10 atau 11** (64-bit)
- **Python 3.11** (Wajib)

### Instalasi

1.  **Clone atau Unduh** repositori ini.
2.  **Jalankan skrip penyetelan otomatis**:
    ```powershell
    python run.py
    ```
    *Catatan: Skrip akan memeriksa Environment Anda dan memberi tahu apa yang harus dilakukan jika ada dependensi yang hilang.*

### Pengaturan Manual (jika diperlukan)

Jika Anda lebih suka mengatur secara manual:

1.  **Buat Virtual Environment**:
    ```powershell
    py -3.11 -m venv venv311
    .\venv311\Scripts\activate
    ```

2.  **Instal Dependensi**:
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi**:
    ```powershell
    python run.py
    ```

## 🖥️ Panduan Penggunaan

1.  **Pilih Kamera**: Gunakan daftar drop-down untuk memilih kamera input Anda.
2.  **Pilih Model**:
    *   **Nano (n)**: Tercepat, direkomendasikan untuk sebagian besar PC.
    *   **Kecil (s)**: Lebih akurat, tetapi membutuhkan CPU yang kuat.
3.  **Kontrol**:
    *   **Mulai**: Memulai deteksi AI.
    *   **Berhenti**: Menjeda deteksi (pratinjau kamera tetap aktif).
    *   **Rekam**: Mengalihkan perekaman video ke folder output.
    *   **Ikon Folder**: Membuka direktori tempat rekaman disimpan.

## 🛠️ Membangun Executable

Untuk membuat file `.exe` mandiri agar mudah didistribusikan:

```powershell
.\venv311\Scripts\activate
python build.py
```

File output `HumanDetectionApp.exe` akan muncul di folder `dist`.

## 📁 Struktur Proyek

Untuk penjelasan rinci tentang basis kode dan cara kerjanya, silakan lihat [CODE_OVERVIEW.md](CODE_OVERVIEW.md).

## ❓ Pemecahan Masalah

| Masalah | Solusi |
|-------|----------|
| **"No module named..."** | Pastikan Anda mengaktifkan virtual environment (`.\venv311\Scripts\activate`). |
| **Kamera tidak ditemukan** | Klik tombol **"R"** (Refresh). Periksa jika aplikasi lain (Zoom/Teams) sedang menggunakan kamera. |
| **FPS Rendah / Lag** | Beralih ke model **Nano** (misalnya, `YOLOv8n`). Pastikan laptop Anda terhubung ke daya. |

## ⚖️ Lisensi
Lisensi MIT
