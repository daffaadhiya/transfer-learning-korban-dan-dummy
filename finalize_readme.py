"""Isi otomatis bagian hasil di README.md dari file di results/ (dipanggil di akhir notebook).
Hanya mengisi ANGKA. Analisis tetap harus Anda baca dan sesuaikan sendiri."""
import json
import re
from collections import Counter
from pathlib import Path

from common import ROOT, load_classes

res = ROOT / "results"
names = load_classes()


def block(tag, text):
    global readme
    pat = re.compile(rf"(<!--{tag}-->).*?(<!--/{tag}-->)", re.S)
    readme = pat.sub(lambda m: f"{m.group(1)}\n{text.strip()}\n{m.group(2)}", readme)


def count(split):
    c, imgs = Counter(), 0
    for f in (ROOT / "dataset" / "labels" / split).glob("*.txt"):
        imgs += 1
        for l in f.read_text().splitlines():
            if l.strip():
                c[int(l.split()[0])] += 1
    return c, imgs


readme = (ROOT / "README.md").read_text(encoding="utf-8")

# 1. statistik data
tr, n_tr = count("train")
va, n_va = count("val")
rows = ["| Kelas | Instance train | Instance val |", "|---|---|---|"]
rows += [f"| {n} | {tr[i]} | {va[i]} |" for i, n in enumerate(names)]
rows.append(f"\nCitra: train = {n_tr}, val = {n_va}.")
block("DATA", "\n".join(rows))

# 2. tabel 3 mode
summ = {}
for m in ("feature", "partial", "scratch"):
    p = res / f"{m}_summary.json"
    if p.exists():
        summ[m] = json.loads(p.read_text())
if (res / "ringkasan.md").exists():
    block("HASIL", (res / "ringkasan.md").read_text())

# 3. latensi
lat = res / "latensi.txt"
if lat.exists():
    block("LATENSI", "```\n" + lat.read_text().strip() + "\n```")

# 4. draf analisis (hanya mengutip angka)
if summ:
    best = max(summ, key=lambda k: summ[k]["best_map50_95"])
    lines = [f"- Mode dengan mAP50-95 tertinggi pada data validasi ini: **{best}** "
             f"(mAP50 = {summ[best]['best_map50']:.3f}, mAP50-95 = {summ[best]['best_map50_95']:.3f})."]
    if "scratch" in summ and best != "scratch":
        d = summ[best]["best_map50_95"] - summ["scratch"]["best_map50_95"]
        if d > 0.02:
            ket = "Ini sejalan dengan hipotesis bahwa transfer learning membantu pada data kecil."
        else:
            ket = "Selisihnya kecil, sehingga belum cukup kuat untuk menyimpulkan manfaat transfer learning."
        lines.append(f"- Dibanding scratch (mAP50-95 = {summ['scratch']['best_map50_95']:.3f}), "
                     f"selisih mAP50-95 = {d:+.3f}. {ket}")
    elif "scratch" in summ:
        lines.append("- Scratch justru terbaik pada data ini. Periksa kemungkinan data validasi terlalu kecil "
                     "atau domain data sumber berbeda jauh dari COCO.")
    t = {m: s["waktu_latih_s"] for m, s in summ.items()}
    lines.append("- Waktu latih (detik): " + ", ".join(f"{m} = {v}" for m, v in t.items()) + ".")
    ap_csv = res / f"{best}_per_class_ap.csv"
    if ap_csv.exists():
        import pandas as pd
        d = pd.read_csv(ap_csv)
        lines.append("- AP50-95 per kelas (model terbaik): " +
                     ", ".join(f"{r.kelas} = {r['AP50-95']:.3f}" for _, r in d.iterrows()) + ".")
    block("ANALISIS", "\n".join(lines))

(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README.md diperbarui.")
