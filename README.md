# Deteksi Korban & Dummy — Robot SAR KRSRI (Transfer Learning, MobileNetV3-Large)

Tugas RET503 Pertemuan 3 — Transfer Learning dan Fine-Tuning Model Visi.
Detektor: **SSDLite320 + MobileNetV3-Large** (torchvision) dengan bobot awal **COCO**, dilatih dengan tiga
pendekatan (feature extraction, fine-tuning parsial, scratch) lalu dibandingkan.

> Kelompok / nama: `[ISI]`

## 1. Ringkasan
- **Tugas:** mendeteksi dan melokalisasi `korban` dan `dummy` pada citra untuk robot SAR KRSRI.
- **Kelas (2):** `korban`, `dummy` (lihat `classes.txt`).
- **Model:** MobileNetV3-Large sebagai backbone detektor SSDLite320 (ringan, cocok untuk edge/robot).
- **Data:** dataset publik Roboflow Universe, ±50 instance per kelas (lihat bagian 2).

## 2. Data dan sumbernya
Data tidak diambil dari kamera robot sendiri, melainkan dari dua dataset publik berlisensi **CC BY 4.0**:

| Kelas | Dataset | Catatan |
|---|---|---|
| `korban` | [Search & Rescue — Trash](https://universe.roboflow.com/trash-32e51/search-rescue) | Foto orang dengan berbagai pose (berbaring, merangkak, duduk, berdiri, terhalang sebagian). Semua pose dipetakan ke `korban`. |
| `dummy` | [Dummy Detection — YoloDetection](https://universe.roboflow.com/yolodetection-q4wg3/dummy-detection-eufpu) | Gambar dummy dengan 1 kelas. |

Setiap gambar tercatat asal, URL, dan lisensinya di `dataset_raw/sumber_publik.csv` dan `dataset_raw/metadata.csv`.
Dari tiap dataset diambil maksimal 50 instance per kelas (`import_roboflow.py --maks-per-kelas 50`).
Pembagian train/val dilakukan **per kelompok gambar berurutan (sesi)**, bukan acak per gambar, untuk mengurangi
kebocoran data antara train dan val.

<!--DATA-->
| Kelas | Instance train | Instance val |
|---|---|---|
| korban | 40 | 10 |
| dummy | 40 | 10 |

Citra: train = 61, val = 18.
<!--/DATA-->

## 3. Metode transfer learning

| Mode | Bobot awal | Yang dilatih | LR |
|---|---|---|---|
| feature | COCO | head baru (klasifikasi + regresi box) | 1e-3 |
| partial | COCO | 4 blok terakhir backbone + extra layers + head | 1e-4 (backbone) / 1e-3 (head) |
| scratch | acak | semua | 1e-3 |

- Head klasifikasi COCO (91 kelas) dibuang dan diganti head baru untuk `2 kelas + background`.
- Lapisan beku tetap `eval()` agar statistik BatchNorm COCO tidak berubah oleh batch kecil.
- Optimizer Adam, `CosineAnnealingLR`, 40 epoch, batch 8, input 320×320.
- Augmentasi: photometric (cahaya), zoom-out, IoU-crop, flip horizontal.
- Normalisasi dilakukan di dalam model torchvision, jadi preprocessing di robot cukup:
  undistort → resize 320×320 → RGB → skala 0–1.

## 4. Hasil

<!--HASIL-->
| Mode    |   mAP50 terbaik |   mAP50-95 terbaik |   Waktu latih (s) | Epoch mAP50>=0.5   |   Param dilatih (M) |
|:--------|----------------:|-------------------:|------------------:|:-------------------|--------------------:|
| feature |           0.418 |              0.188 |               256 | tidak tercapai     |                0.15 |
| partial |           0.496 |              0.248 |               320 | tidak tercapai     |                2.06 |
| scratch |           0.009 |              0.001 |               402 | tidak tercapai     |                2.22 |
<!--/HASIL-->

Grafik: `results/kurva_3_mode.png` · AP per kelas: `results/*_per_class_ap.csv` · Confusion matrix: `results/*_confusion.png`

### Latensi (pipeline lengkap, batas 15 FPS = 67 ms/frame)
Diukur di **CPU Colab/PC, bukan di robot** (robot tidak tersedia). Angka di robot (Raspberry Pi/Jetson) akan berbeda
dan perlu diukur ulang dengan `latency.py` saat perangkatnya ada.

<!--LATENSI-->
```
[info] calib.npz tidak dipakai (undistort dilewati)

Latensi per frame (30 run, device=cpu):
  akuisisi+undistort   rata-rata     1.8 ms   p95     1.9 ms
  preprocess           rata-rata     0.5 ms   p95     0.6 ms
  inferensi            rata-rata    35.1 ms   p95    38.7 ms
  postprocess          rata-rata     0.1 ms   p95     0.1 ms
  total                rata-rata    37.3 ms   p95    41.2 ms

FPS (rata-rata, pipeline lengkap): 26.8
Anggaran 15 FPS = 67 ms -> MEMENUHI
Catatan: latensi ROS2 (publish/subscribe) belum termasuk; ukur terpisah di robot.
```
<!--/LATENSI-->

## 5. Analisis

Draf otomatis dari angka (periksa dan lengkapi sendiri):

<!--ANALISIS-->
- Mode dengan mAP50-95 tertinggi pada data validasi ini: **partial** (mAP50 = 0.496, mAP50-95 = 0.248).
- Dibanding scratch (mAP50-95 = 0.001), selisih mAP50-95 = +0.247. Ini sejalan dengan hipotesis bahwa transfer learning membantu pada data kecil.
- Waktu latih (detik): feature = 256, partial = 320, scratch = 402.
- AP50-95 per kelas (model terbaik): dummy = 0.069, korban = 0.426.
<!--/ANALISIS-->

Catatan analisis (isi / sesuaikan):
1. **Apakah TL membantu?** Bandingkan mode feature/partial dengan scratch pada tabel di atas.
2. **Kelas terlemah dan kesalahan umum:** lihat `*_per_class_ap.csv` dan `*_confusion.png`.
3. **Pengaruh data:** data kecil (±50 instance/kelas) dan validasi yang kecil membuat angka mAP tidak stabil.
4. **Negative transfer / leakage:** split per sesi; mAP validasi yang terlalu tinggi perlu dicurigai.

## 6. Keterbatasan (jujur)
- Data berasal dari dataset publik, **bukan dari kamera dan arena robot KRSRI**, sehingga performa di arena nyata belum terbukti.
- Kelas `korban` dilatih dari foto orang dengan berbagai pose sebagai pendekatan, bukan dari boneka korban lomba.
- Jumlah data kecil (±50 instance per kelas); validasi hanya puluhan gambar sehingga hasil sangat bergantung pada pembagian data.
- Latensi belum diukur di perangkat robot.
- Rencana lanjutan: ambil data dengan kamera robot (`capture.py`), anotasi sendiri, dan latih ulang.

## 7. Cara mengulang
**Cara termudah (lokal, satu perintah):** unduh dua dataset dari Roboflow dalam format YOLOv8, simpan sebagai
`data_publik/korban.zip` dan `data_publik/dummy.zip`, lalu:
```bash
pip install -r requirements.txt
python run_all.py            # impor -> split -> latih 3 mode -> evaluasi -> latensi -> isi README
```
Alternatif dengan GPU gratis: buka `Jalankan_di_Colab.ipynb` di Google Colab dan jalankan semua sel.

Cara manual per langkah:
```bash
python import_roboflow.py --src <folder_dataset_korban> --tag kor1 --semua-ke korban --maks-per-kelas 50
python import_roboflow.py --src <folder_dataset_dummy>  --tag dum1 --semua-ke dummy  --maks-per-kelas 50
python split.py --val-ratio 0.2
python check_dataset.py --target 50
python train.py --mode feature && python train.py --mode partial && python train.py --mode scratch
python plot_results.py
python evaluate.py --weights weights/partial_best.pt
python latency.py --weights weights/partial_best.pt --image <contoh.png>
python finalize_readme.py
```

## 8. Struktur repositori
```
classes.txt  import_roboflow.py  split.py  check_dataset.py  dataset.py  common.py
train.py  evaluate.py  latency.py  plot_results.py  finalize_readme.py  run_all.py  requirements.txt
Jalankan_di_Colab.ipynb   docs/desain.md   docs/panduan_label.md
data_publik/ (zip unduhan, tidak diunggah)   dataset_raw/ (images, labels, metadata.csv, sumber_publik.csv)   dataset/ (hasil split)
results/   weights/
```

## 9. Sitasi data
- Trash. *Search & Rescue Dataset*. Roboflow Universe, 2026. https://universe.roboflow.com/trash-32e51/search-rescue (CC BY 4.0)
- YoloDetection. *Dummy Detection Dataset*. Roboflow Universe, 2025. https://universe.roboflow.com/yolodetection-q4wg3/dummy-detection-eufpu (CC BY 4.0)
