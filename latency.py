"""Ukur latensi pipeline LENGKAP per frame (anggaran 15 FPS = ~67 ms, slide 16).

Jalankan di perangkat robot (Raspberry Pi / Jetson), bukan di laptop:
    python latency.py --weights weights/partial_best.pt --image contoh.png --runs 100
Tahapan: baca/undistort -> preprocess -> inferensi -> postprocess.
"""
import argparse
import time

import cv2
import numpy as np
import torch

from common import build_model, load_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True, help="citra contoh (resolusi sama dgn kamera robot)")
    ap.add_argument("--calib", default="calib.npz")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--score", type=float, default=0.4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ck = torch.load(a.weights, map_location="cpu")
    names = ck.get("classes") or load_classes()
    model = build_model(len(names), "scratch")      # struktur saja; bobot dimuat di bawah
    model.load_state_dict(ck["state_dict"])
    model.to(a.device).eval()

    K = dist = None
    try:
        d = np.load(a.calib)
        K = next(d[k] for k in ("K", "mtx", "camera_matrix") if k in d)
        dist = next(d[k] for k in ("dist", "D", "dist_coeffs") if k in d)
    except Exception:
        print("[info] calib.npz tidak dipakai (undistort dilewati)")

    t = {"akuisisi+undistort": [], "preprocess": [], "inferensi": [], "postprocess": [], "total": []}
    with torch.inference_mode():
        for i in range(a.runs + 10):                 # 10 run pertama = warm-up
            t0 = time.perf_counter()
            frame = cv2.imread(a.image)
            if K is not None:
                frame = cv2.undistort(frame, K, dist)
            t1 = time.perf_counter()
            x = cv2.cvtColor(cv2.resize(frame, (320, 320)), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(x).permute(2, 0, 1).float().div(255).to(a.device)
            t2 = time.perf_counter()
            out = model([x])[0]
            if a.device != "cpu":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            keep = out["scores"] >= a.score
            _ = (out["boxes"][keep].cpu().numpy(), out["labels"][keep].cpu().numpy())
            t4 = time.perf_counter()
            if i >= 10:
                for k, v in zip(t, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                    t[k].append(v * 1000)

    print(f"\nLatensi per frame ({a.runs} run, device={a.device}):")
    for k, v in t.items():
        print(f"  {k:<20} rata-rata {np.mean(v):7.1f} ms   p95 {np.percentile(v,95):7.1f} ms")
    mean = np.mean(t["total"])
    print(f"\nFPS (rata-rata, pipeline lengkap): {1000/mean:.1f}")
    print("Anggaran 15 FPS = 67 ms ->", "MEMENUHI" if mean <= 67 else "BELUM MEMENUHI")
    print("Catatan: latensi ROS2 (publish/subscribe) belum termasuk; ukur terpisah di robot.")


if __name__ == "__main__":
    main()
