"""Latih detektor SSDLite320 + MobileNetV3-Large dengan 3 pendekatan (eksperimen P2).

    python train.py --mode feature     # backbone beku, hanya head
    python train.py --mode partial     # + blok akhir backbone dibuka (LR berbeda)
    python train.py --mode scratch     # tanpa pretrained, semua dilatih
"""
import argparse
import csv
import json
import time

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from common import ROOT, build_model, count_params, load_classes, param_groups, set_train_mode
from dataset import YoloDetectionDataset, collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        preds = model(imgs)
        metric.update(
            [{k: v.cpu() for k, v in p.items()} for p in preds],
            [{k: v.cpu() for k, v in t.items()} for t in tgts],
        )
    r = metric.compute()
    return float(r["map_50"]), float(r["map"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["feature", "partial", "scratch"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--unfreeze-blocks", type=int, default=4)
    ap.add_argument("--target-map50", type=float, default=0.5,
                    help="catat epoch pertama saat mAP50 >= nilai ini")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-flip", action="store_true",
                    help="matikan flip horizontal bila kelas dibedakan kiri/kanan")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = load_classes()
    data = ROOT / "dataset"

    tr_ds = YoloDetectionDataset(data, "train", train=True, flip=not a.no_flip)
    va_ds = YoloDetectionDataset(data, "val", train=False)
    tr = DataLoader(tr_ds, a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, drop_last=len(tr_ds) > a.batch)
    va = DataLoader(va_ds, a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)

    model = build_model(len(names), a.mode, a.unfreeze_blocks).to(device)
    total, trainable = count_params(model)
    print(f"mode={a.mode} | kelas={len(names)} | train={len(tr_ds)} val={len(va_ds)} | "
          f"parameter total={total/1e6:.2f}M dilatih={trainable/1e6:.2f}M | device={device}")

    opt = torch.optim.Adam(param_groups(model, a.mode, a.lr_head, a.lr_backbone))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "weights").mkdir(exist_ok=True)
    log_path = ROOT / "results" / f"{a.mode}_log.csv"
    best50 = best5095 = 0.0
    epoch_target = None
    t0 = time.time()

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "map50", "map50_95", "waktu_s"])
        for ep in range(1, a.epochs + 1):
            set_train_mode(model, a.mode)
            losses = []
            for imgs, tgts in tqdm(tr, desc=f"epoch {ep}/{a.epochs}", leave=False):
                imgs = [i.to(device) for i in imgs]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
                loss = sum(model(imgs, tgts).values())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                losses.append(loss.item())
            sched.step()

            m50, m5095 = evaluate(model, va, device)
            tl = sum(losses) / max(len(losses), 1)
            w.writerow([ep, f"{tl:.4f}", f"{m50:.4f}", f"{m5095:.4f}", f"{time.time()-t0:.0f}"])
            f.flush()
            print(f"epoch {ep:>3} | loss {tl:.3f} | mAP50 {m50:.3f} | mAP50-95 {m5095:.3f}")
            if epoch_target is None and m50 >= a.target_map50:
                epoch_target = ep
            if m5095 >= best5095:
                best50, best5095 = m50, m5095
                torch.save({"state_dict": model.state_dict(), "classes": names,
                            "mode": a.mode, "unfreeze_blocks": a.unfreeze_blocks},
                           ROOT / "weights" / f"{a.mode}_best.pt")

    summary = {
        "mode": a.mode, "epochs": a.epochs, "best_map50": best50, "best_map50_95": best5095,
        "waktu_latih_s": round(time.time() - t0), "epoch_map50_ke_target": epoch_target,
        "target_map50": a.target_map50, "param_total": total, "param_dilatih": trainable,
        "n_train": len(tr_ds), "n_val": len(va_ds), "n_kelas": len(names),
    }
    (ROOT / "results" / f"{a.mode}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
