"""Bagi dataset_raw -> dataset/{train,val} BERDASARKAN SESI (mencegah data leakage, slide 22).

Syarat: setiap gambar di dataset_raw/images punya label YOLO di dataset_raw/labels/<nama>.txt
(gambar tanpa file label dianggap citra negatif / arena kosong).
Sesi diambil dari awalan nama file sebelum '_' pertama (hasil capture.py).

    python split.py --val-ratio 0.2 --seed 0
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import ROOT, load_classes

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def read_classes_in(lbl):
    s = set()
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if line.strip():
                s.add(int(line.split()[0]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = ROOT / "dataset_raw"
    out = ROOT / "dataset"
    names = load_classes()
    nc = len(names)

    sessions = defaultdict(list)
    for p in sorted((raw / "images").iterdir()):
        if p.suffix.lower() in IMG_EXT:
            sessions[p.name.split("_")[0]].append(p)
    if len(sessions) < 2:
        raise SystemExit("Butuh minimal 2 sesi pengambilan agar bisa dipisah train/val.")

    sess_cls = {s: set().union(*(read_classes_in(raw / "labels" / (p.stem + ".txt")) for p in ps))
                for s, ps in sessions.items()}
    total = sum(len(v) for v in sessions.values())

    # pilih sesi val secara acak, tetapi sampai tiap kelas punya ~val_ratio dari total instance-nya
    inst = {s_: [0] * nc for s_ in sessions}
    for s_, ps in sessions.items():
        for p in ps:
            lbl = raw / "labels" / (p.stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    if line.strip():
                        inst[s_][int(line.split()[0])] += 1
    tot = [sum(inst[s_][c] for s_ in sessions) for c in range(nc)]
    need = [max(1, round(a.val_ratio * t)) if t else 0 for t in tot]
    order = list(sessions)
    random.Random(a.seed).shuffle(order)
    val, got = [], [0] * nc
    for s_ in order:
        if all(got[c] >= need[c] for c in range(nc)):
            break
        if any(got[c] < need[c] and inst[s_][c] > 0 for c in range(nc)):
            val.append(s_)
            got = [got[c] + inst[s_][c] for c in range(nc)]
    train = [s_ for s_ in order if s_ not in val]
    if not train:                      # jaga agar train tidak kosong
        train, val = val[1:], val[:1]

    def cover(ss):
        return set().union(*(sess_cls[s] for s in ss)) if ss else set()

    # kelas yang belum ada di val: pindahkan sesi dari train yang memuatnya (selama train tetap berisi)
    for c in range(nc):
        if c not in cover(val):
            for s in train:
                if c in sess_cls[s] and len(train) > 1:
                    train.remove(s)
                    val.append(s)
                    break

    miss_val = [names[c] for c in range(nc) if c not in cover(val)]
    miss_train = [names[c] for c in range(nc) if c not in cover(train)]

    for split, ss in (("train", train), ("val", val)):
        for sub in ("images", "labels"):
            d = out / sub / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.name != ".gitkeep":
                    f.unlink()
        for s in ss:
            for p in sessions[s]:
                shutil.copy2(p, out / "images" / split / p.name)
                lbl = raw / "labels" / (p.stem + ".txt")
                dst = out / "labels" / split / (p.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(lbl, dst)
                else:
                    dst.write_text("")      # citra negatif

    n_tr = sum(len(sessions[s]) for s in train)
    n_va = sum(len(sessions[s]) for s in val)
    print(f"Sesi train: {sorted(train)}  ({n_tr} citra)")
    print(f"Sesi val  : {sorted(val)}  ({n_va} citra)")
    if miss_val:
        print(f"[PERINGATAN] {len(miss_val)} kelas TIDAK ada di val: {miss_val[:10]}...")
    if miss_train:
        print(f"[PERINGATAN] {len(miss_train)} kelas TIDAK ada di train: {miss_train[:10]}...")
    print("Tips: jika ada kelas hilang, tambah sesi pengambilan yang memuat kelas tersebut.")


if __name__ == "__main__":
    main()
