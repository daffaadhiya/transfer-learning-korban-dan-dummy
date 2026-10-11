"""Jalankan SELURUH pipeline dengan satu perintah.

Persiapan: taruh dua hasil unduhan Roboflow (format YOLOv8) di folder data_publik/ :
    data_publik/korban.zip   (dataset search-rescue)
    data_publik/dummy.zip    (dataset dummy-detection)
(boleh juga folder hasil ekstrak, atau pakai --korban / --dummy untuk menunjuk lokasi lain)

Lalu:   python run_all.py

Urutan: impor data (maks 50 kotak/kelas) -> split per sesi -> cek data -> latih 3 mode
        -> rekap & grafik -> AP per kelas + confusion matrix -> latensi (laptop) -> isi README.
Tambah --epochs 5 untuk uji cepat.
"""
import argparse
import json
import shutil
import zlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from common import ROOT

PUB = ROOT / "data_publik"
RAW = ROOT / "dataset_raw"

SUMBER = {
    "korban": {
        "url": "https://universe.roboflow.com/trash-32e51/search-rescue",
        "nama": "Search & Rescue (oleh Trash), Roboflow Universe",
    },
    "dummy": {
        "url": "https://universe.roboflow.com/yolodetection-q4wg3/dummy-detection-eufpu",
        "nama": "Dummy Detection (oleh YoloDetection), Roboflow Universe",
    },
}


def run(cmd, judul, simpan=None):
    print(f"\n{'=' * 8} {judul} {'=' * 8}", flush=True)
    if simpan:
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        print(r.stdout + r.stderr)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / simpan).write_text(r.stdout, encoding="utf-8")
        out = r.stdout
    else:
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        out = ""
    if r.returncode != 0:
        raise SystemExit(f"\n[GAGAL] Langkah '{judul}' berhenti dengan error. Baca pesan di atas, lalu jalankan lagi.")
    return out


def ekstrak_aman(zip_path, out, batas=0.4):
    """Ekstrak zip file demi file. File yang rusak (CRC salah/terpotong) dilewati, bukan menghentikan program."""
    rusak = []
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        for info in infos:
            try:
                z.extract(info, out)
            except (zipfile.BadZipFile, zlib.error, EOFError, OSError):
                rusak.append(info.filename)
                f = Path(out) / info.filename
                if f.exists():
                    f.unlink()            # buang file setengah jadi
    if infos and len(rusak) > batas * len(infos):
        raise SystemExit(f"Terlalu banyak file rusak ({len(rusak)} dari {len(infos)}, batas {batas:.0%}) di "
                         f"{Path(zip_path).name}. Unduh ulang zip-nya, lebih baik versi dataset yang lebih kecil.")
    return rusak


def cari_sumber(nilai, nama, tmp, batas_rusak=0.4):
    """Terima zip atau folder; kembalikan folder yang berisi data.yaml."""
    p = Path(nilai) if nilai else None
    if p is None:
        for kandidat in (PUB / f"{nama}.zip", PUB / nama):
            if kandidat.exists():
                p = kandidat
                break
    if p is None or not p.exists():
        raise SystemExit(f"Data '{nama}' tidak ditemukan. Taruh {PUB / (nama + '.zip')} atau pakai --{nama} <lokasi>.")
    if p.suffix.lower() == ".zip":
        out = Path(tmp) / nama                      # folder sementara berpath pendek (hindari batas 260 karakter Windows)
        rusak = ekstrak_aman(p, out, batas_rusak)
        if rusak:
            print(f"[PERINGATAN] {len(rusak)} file di {p.name} rusak dan dilewati "
                  f"(mis. {rusak[0]}). Sisanya tetap dipakai.")
        p = out
    hasil = list(p.rglob("data.yaml"))
    if not hasil:
        raise SystemExit(f"data.yaml tidak ada di '{p}'. Unduh ulang dengan format YOLOv8.")
    return hasil[0].parent


def siapkan_raw():
    ada_impor = (RAW / "sumber_publik.csv").exists()
    ada_isi = any((RAW / "images").glob("*")) if (RAW / "images").exists() else False
    if ada_isi and not ada_impor:
        raise SystemExit("dataset_raw/ berisi foto milik Anda sendiri. Pindahkan dulu, atau hapus isinya, agar tidak tertimpa.")
    for sub in ("images", "labels"):
        d = RAW / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.name != ".gitkeep":
                f.unlink()
    for f in ("metadata.csv", "sumber_publik.csv"):
        (RAW / f).unlink(missing_ok=True)


def baca_summary(mode):
    p = ROOT / "results" / f"{mode}_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--korban", default="", help="zip/folder dataset korban (default: data_publik/korban.zip)")
    ap.add_argument("--dummy", default="", help="zip/folder dataset dummy (default: data_publik/dummy.zip)")
    ap.add_argument("--per-kelas", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--maks-rusak", type=float, default=0.4,
                    help="batas persentase file rusak di zip yang masih ditoleransi (0.4 = 40%%)")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="krsri_")
    try:
        src = {n: cari_sumber(v, n, tmp, a.maks_rusak) for n, v in (("korban", a.korban), ("dummy", a.dummy))}
        siapkan_raw()
        for n in ("korban", "dummy"):
            run(["import_roboflow.py", "--src", str(src[n]), "--tag", n[:3] + "1", "--map-semua", n,
                 "--maks-per-kelas", str(a.per_kelas), "--url", SUMBER[n]["url"], "--lisensi", "CC BY 4.0"],
                f"Impor data '{n}'")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    run(["split.py", "--val-ratio", str(a.val_ratio)], "Bagi train/val per sesi")
    stat = run(["check_dataset.py", "--target", str(a.per_kelas)], "Cek jumlah data per kelas", "dataset_stats.txt")

    for mode in ("feature", "partial", "scratch"):
        run(["train.py", "--mode", mode, "--epochs", str(a.epochs), "--batch", str(a.batch), "--workers", "0"],
            f"Latih mode {mode}")

    run(["plot_results.py"], "Rekap 3 mode")
    S = {m: baca_summary(m) for m in ("feature", "partial", "scratch")}
    terbaik = max(S, key=lambda m: S[m]["best_map50_95"] if S[m] else -1)
    print(f"\nMode terbaik (mAP50-95): {terbaik}")
    run(["evaluate.py", "--weights", f"weights/{terbaik}_best.pt", "--device", "cpu"], f"Evaluasi per kelas ({terbaik})",
        "evaluasi_per_kelas.txt")

    contoh = sorted((ROOT / "dataset" / "images" / "val").glob("*"))
    contoh = [p for p in contoh if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    lat = ""
    if contoh:
        lat = run(["latency.py", "--weights", f"weights/{terbaik}_best.pt", "--image", str(contoh[0]), "--runs", "30"],
                  "Latensi (CPU laptop, bukan robot)", "latensi.txt")

    run(["finalize_readme.py"], "Isi README dengan hasil")
    print("\nSELESAI. Hasil ada di folder results/ dan sudah dimasukkan ke README.md.")
    print("Berikutnya: lengkapi nama kelompok di README/docs/desain.md, lalu push ke GitHub.")


if __name__ == "__main__":
    main()
