"""Ambil citra dari kamera robot, terkoreksi distorsi (calib.npz dari pertemuan 2).

Contoh:
    python capture.py --sesi s01 --cahaya terang --jarak sedang
SPASI = simpan frame, Q = keluar.
Hasil: dataset_raw/images/<sesi>_<cahaya>_<jarak>_<nnn>.png dan baris di dataset_raw/metadata.csv
"""
import argparse
import csv
from datetime import date
from pathlib import Path

import cv2
import numpy as np

RAW = Path(__file__).resolve().parent / "dataset_raw"


def load_calib(path):
    d = np.load(path)
    K = next(d[k] for k in ("K", "mtx", "camera_matrix") if k in d)
    dist = next(d[k] for k in ("dist", "D", "dist_coeffs") if k in d)
    return K, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesi", required=True, help="ID sesi pengambilan (dipakai untuk split)")
    ap.add_argument("--cahaya", required=True, help="terang/redup/jendela/bayangan")
    ap.add_argument("--jarak", default="sedang", help="dekat/sedang/jauh")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--calib", default="calib.npz")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()

    (RAW / "images").mkdir(parents=True, exist_ok=True)
    meta = RAW / "metadata.csv"
    new = not meta.exists()

    K = dist = None
    if Path(a.calib).exists():
        K, dist = load_calib(a.calib)
    else:
        print(f"[PERINGATAN] {a.calib} tidak ditemukan -> citra TIDAK di-undistort")

    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)   # samakan dengan resolusi saat kalibrasi!
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    prefix = f"{a.sesi}_{a.cahaya}_{a.jarak}"
    n = len(list((RAW / "images").glob(prefix + "_*.png")))

    with open(meta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["nama_file", "sesi", "tanggal", "kondisi_cahaya", "jarak"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if K is not None:
                frame = cv2.undistort(frame, K, dist)
            view = frame.copy()
            cv2.putText(view, f"{prefix} tersimpan: {n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("capture (SPASI=simpan, Q=keluar)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                n += 1
                name = f"{prefix}_{n:03d}.png"
                cv2.imwrite(str(RAW / "images" / name), frame)
                w.writerow([name, a.sesi, date.today().isoformat(), a.cahaya, a.jarak])
                f.flush()
            elif key in (ord("q"), 27):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
