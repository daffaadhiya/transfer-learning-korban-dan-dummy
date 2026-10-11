"""AP per kelas + confusion matrix pada data val (bahan analisis README, minggu 8 & 12).

    python evaluate.py --weights weights/partial_best.pt
Keluaran: results/<mode>_per_class_ap.csv, results/<mode>_confusion.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from common import ROOT, build_model
from dataset import YoloDetectionDataset, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.weights, map_location="cpu")
    names, mode = ck["classes"], ck["mode"]
    nc = len(names)
    model = build_model(nc, "scratch")
    model.load_state_dict(ck["state_dict"])
    model.to(a.device).eval()

    ds = YoloDetectionDataset(ROOT / "dataset", "val")
    dl = DataLoader(ds, 8, shuffle=False, collate_fn=collate)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    cm = np.zeros((nc + 1, nc + 1), dtype=int)       # baris = GT, kolom = prediksi; index nc = background

    with torch.no_grad():
        for imgs, tgts in dl:
            preds = model([i.to(a.device) for i in imgs])
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            metric.update(preds, tgts)
            for p, t in zip(preds, tgts):
                keep = p["scores"] >= a.score
                pb, pl = p["boxes"][keep], p["labels"][keep] - 1
                gb, gl = t["boxes"], t["labels"] - 1
                used = set()
                if len(gb) and len(pb):
                    iou = box_iou(gb, pb)
                    for g in range(len(gb)):
                        j = int(iou[g].argmax())
                        if iou[g, j] >= a.iou and j not in used:
                            cm[int(gl[g]), int(pl[j])] += 1
                            used.add(j)
                        else:
                            cm[int(gl[g]), nc] += 1      # terlewat
                else:
                    for g in gl:
                        cm[int(g), nc] += 1
                for j in range(len(pb)):
                    if j not in used:
                        cm[nc, int(pl[j])] += 1          # false positive

    r = metric.compute()
    classes = r["classes"].tolist()
    ap_cls = r["map_per_class"].tolist()
    rows = [{"kelas": names[c - 1], "AP50-95": round(v, 4)} for c, v in zip(classes, ap_cls)]
    df = pd.DataFrame(rows).sort_values("AP50-95")
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"{mode}_per_class_ap.csv", index=False)
    print(f"mAP50={float(r['map_50']):.3f}  mAP50-95={float(r['map']):.3f}")
    print("10 kelas terlemah:\n", df.head(10).to_string(index=False))

    # pasangan kelas yang paling sering tertukar
    off = cm[:nc, :nc].copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:10]
    print("\nPasangan sering tertukar (GT -> prediksi):")
    for g, p in idx:
        if off[g, p] > 0:
            print(f"  {names[g]} -> {names[p]}: {off[g, p]}x")

    fig, ax = plt.subplots(figsize=(max(8, nc * 0.18), max(8, nc * 0.18)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion matrix ({mode}), baris/kolom terakhir = background")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"{mode}_confusion.png", dpi=150)


if __name__ == "__main__":
    main()
