"""Hitung jumlah instance per kelas (train & val) dan tandai kelas yang < target (default 50)."""
import argparse
from collections import Counter

from common import ROOT, load_classes


def count(split):
    c, imgs, neg = Counter(), 0, 0
    d = ROOT / "dataset" / "labels" / split
    for f in d.glob("*.txt"):
        imgs += 1
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        neg += not lines
        for l in lines:
            c[int(l.split()[0])] += 1
    return c, imgs, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50, help="min. instance per kelas (train+val)")
    a = ap.parse_args()
    names = load_classes()
    tr, n_tr, neg_tr = count("train")
    va, n_va, neg_va = count("val")
    print(f"Citra: train={n_tr} (negatif {neg_tr}), val={n_va} (negatif {neg_va})")
    print(f"{'kelas':<22}{'train':>7}{'val':>6}{'total':>7}")
    low = []
    for i, n in enumerate(names):
        t = tr[i] + va[i]
        flag = "" if t >= a.target else "  <-- kurang"
        if t < a.target:
            low.append(n)
        print(f"{n:<22}{tr[i]:>7}{va[i]:>6}{t:>7}{flag}")
    unk = [k for k in list(tr) + list(va) if k >= len(names)]
    if unk:
        print(f"[ERROR] ID kelas di luar classes.txt: {sorted(set(unk))}")
    print(f"\n{len(low)}/{len(names)} kelas belum mencapai {a.target} instance.")


if __name__ == "__main__":
    main()
