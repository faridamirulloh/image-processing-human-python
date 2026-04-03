# Ikhtisar Kode (Code Overview)

Halo! Selamat datang di dokumentasi singkat untuk proyek **Aplikasi Deteksi Manusia**. Dokumen ini akan membantu Anda memahami struktur kode dan bagaimana komponen-komponen utama bekerja bersama. Mari kita mulai! 🚀

---

## 📂 Struktur Direktori

Berikut adalah tampilan cepat bagaimana proyek ini diatur:

```
root/
├── build.py                # Skrip untuk membuat file executable (.exe)
├── run.py                  # Skrip untuk menjalankan aplikasi saat development
├── HumanDetectionApp.spec  # Konfigurasi PyInstaller
├── src/                    # Kode sumber utama
│   ├── main.py             # Titik masuk (entry point) aplikasi
│   ├── app.py              # Logika utama UI dan orkestrasi
│   ├── services/           # Logika bisnis dan pemrosesan di balik layar
│   ├── utils/              # Konstanta dan gaya tampilan
│   └── widgets/            # Komponen tampilan UI kustom
```

---

## 🧩 Komponen Utama

Mari kita bahas apa saja yang ada di dalam folder `src/` agar kita lebih paham fungsinya masing-masing.

### 1. Aplikasi Utama (`src/`)

*   **`main.py`**: Ini adalah pintu masuk aplikasi kita. File ini menyiapkan aplikasi PyQt, mengatur skala tampilan agar tajam di layar resolusi tinggi, dan memunculkan jendela utama.
*   **`app.py`**: Ini adalah "otak" dari antarmuka pengguna (UI). Di sini kita mengatur tata letak jendela, menghubungkan tombol-tombol dengan fungsinya, dan mengelola layanan-layanan lain (seperti kamera dan detektor).

### 2. Layanan (`src/services/`)

Bagian ini berisi logika "berat" yang bekerja di belakang layar:

*   **`camera_service.py`**: Bertanggung jawab untuk mencari kamera yang terhubung ke komputer Anda. Ia memastikan kita bisa mendapatkan daftar kamera yang siap digunakan.
*   **`video_service.py`**: Menangani pengambilan gambar dari kamera secara *real-time*. Layanan ini berjalan di *thread* terpisah supaya aplikasi tidak macet saat membaca data kamera.
*   **`detector_service.py`**: Di sinilah kecerdasan buatan (AI) bekerja! Menggunakan model YOLO (v8, v11, atau v12) untuk mendeteksi manusia dalam video. Canggih, kan? 😎
*   **`recording_service.py`**: Mengurus penyimpanan video rekaman dan pengambilan *screenshot* (tangkapan layar) ke folder dokumen Anda.

### 3. Utilitas (`src/utils/`)

Folder ini berisi hal-hal pendukung agar kode lebih rapi:

*   **`constants.py`**: Tempat kita menyimpan pengaturan penting, seperti ukuran jendela default, daftar model YOLO, dan konfigurasi lainnya.
*   **`styles.py`**: Berisi "resep" tampilan (CSS) untuk membuat aplikasi terlihat modern dan menarik.

### 4. Widget (`src/widgets/`)

Komponen tampilan khusus yang kita buat sendiri:

*   **`video_widget.py`**: Area untuk menampilkan video dari kamera. Widget ini pintar menyesuaikan ukuran gambar agar selalu pas dan proporsional.
*   **`stats_widget.py`**: Panel di samping kanan yang menampilkan statistik, seperti jumlah orang yang terdeteksi dan kecepatan pemrosesan (FPS).

---

## 🛠️ Cara Menjalankan

Untuk menjalankan aplikasi saat sedang mengembangkan fitur baru (mode *development*), Anda cukup menjalankan perintah ini di terminal:

```bash
python run.py
```

Jika Anda ingin membuat aplikasi ini menjadi satu file `.exe` yang bisa dijalankan di komputer lain tanpa perlu instal Python, gunakan perintah:

```bash
python build.py
```

---

Semoga panduan ini membantu Anda menjelajahi kode dengan lebih mudah! Jika ada pertanyaan, jangan ragu untuk mengecek komentar di dalam kode ya. Selamat coding! 💻✨
