# Panduan Anotasi

Tahap ini memakai dataset publik yang sudah dianotasi (lihat README). Panduan ini dipakai jika nanti menambah
data dari kamera robot.

Format: YOLO (`kelas cx cy w h`, ternormalisasi 0-1), satu `.txt` per citra di `dataset_raw/labels/`.

| ID | Kelas | Definisi |
|---|---|---|
| 0 | `korban` | Manusia/boneka korban dalam pose apa pun (berbaring, merangkak, duduk, berdiri, sebagian tertutup) |
| 1 | `dummy` | Objek dummy yang diletakkan sebagai target |

Aturan: box mengepas bagian objek yang terlihat; satu box per objek; citra tanpa objek tidak perlu file label.
