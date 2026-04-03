"""
Skrip untuk menjalankan aplikasi dalam mode development
Membutuhkan Python 3.11
"""

import sys
import os

# Periksa versi Python
if sys.version_info[:2] != (3, 11):
    print("=" * 50)
    print("ERROR: Python 3.11 is required!")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
    print("\nPlease set up the environment:")
    print("  py -3.11 -m venv venv311")
    print("  .\\venv311\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print("=" * 50)
    sys.exit(1)

# Tambahkan direktori src ke path Python agar impor berfungsi dengan benar
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from main import main

if __name__ == "__main__":
    main()
