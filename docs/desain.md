# Dokumen Desain Pipeline Persepsi

Kelompok: `[Daffa' Dhiya 'Ulhaq (4222401075)]` · Proyek RET501: Robot SAR KRSRI · Versi 0.1

| Komponen | Isi |
|---|---|
| **Misi proyek** | Robot SAR mendeteksi dan melokalisasi korban dan dummy di arena agar dapat dihampiri/ditandai. |
| **Kelas objek** | 2 kelas: `korban`, `dummy`. |
| **Kamera & dudukan** | `[ISI: resolusi, tinggi, sudut, jarak kerja]` — belum dipakai karena versi ini memakai data publik. |
| **Unit komputasi** | `[ISI: Raspberry Pi / Jetson]` (target deployment); pelatihan dan pengukuran awal di laptop CPU. |
| **Target kinerja** | mAP50 setinggi mungkin pada validasi; ≥ 15 FPS onboard (≈ 67 ms/frame, pipeline lengkap). |
| **Kandidat model** | (1) **SSDLite320 + MobileNetV3-Large** (ringan, bobot COCO tersedia, cocok edge). (2) **YOLO varian n/s** sebagai pembanding pada pertemuan berikutnya. |
| **Strategi TL** | Feature extraction → fine-tuning parsial (LR 1e-4/1e-3); scratch sebagai pembanding. Alasan: data sedikit (±50 instance/kelas) dan domain COCO berdekatan (orang/objek sehari-hari). |
| **Rencana data** | Fase 1: dataset publik CC BY 4.0 (Roboflow Universe), ±50 instance/kelas, dicatat di `sumber_publik.csv`. Fase 2: foto dari kamera robot di arena (variasi jarak, cahaya, latar, occlusion), anotasi bounding box, split per sesi. |

## Risiko dan mitigasi

| # | Risiko | Mitigasi |
|---|---|---|
| 1 | Data publik berbeda domain dari arena KRSRI (kamera, cahaya, bentuk korban) | Ambil data dari robot di fase 2; evaluasi ulang pada data arena |
| 2 | Data sedikit (±50/kelas) → mAP tidak stabil | Augmentasi, tambah data, laporkan keterbatasan, bandingkan 3 mode |
| 3 | Data leakage antar frame yang mirip | Split berdasarkan sesi (`split.py`) |
| 4 | Latensi di robot melebihi 67 ms | Ukur `latency.py` di robot; turunkan resolusi/ganti model; ekspor ONNX/TensorRT |
| 5 | `korban` dari foto orang, bukan boneka korban robot | Tulis sebagai keterbatasan; ganti dengan data robot di fase 2 |
