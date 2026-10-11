"""Impor dataset publik (ekspor Roboflow/YOLO) ke dataset_raw/ agar bisa dipakai split.py & train.py.

Yang dilakukan:
  1. Menggabungkan folder train/valid/test hasil unduhan.
  2. Memetakan nama kelas sumber -> nama kelas di classes.txt (kelas yang tidak dipetakan dibuang).
  3. Memberi nama file baru + membagi gambar menjadi beberapa 'sesi' (potongan berurutan, supaya
     frame yang mirip/hasil augmentasi tetap dalam sesi yang sama -> mencegah data leakage).
  4. Mencatat asal data di dataset_raw/sumber_publik.csv (provenance) dan metadata.csv.

Contoh (dataset punya kelas 'victim' dan 'dummy'):
    python import_roboflow.py --src "D:\\unduhan\\victim-detection-1" --tag vic1 ^
        --map victim:korban --map dummy:dummy --maks-per-kelas 50 ^
        --url https://universe.roboflow.com/... --lisensi "CC BY 4.0"
Lihat nama kelas sumber dulu dengan:  python import_roboflow.py --src <folder> --lihat-kelas
"""
import argparse
import ast
import csv
import re
import shutil
from datetime import date
from pathlib import Path

from common import ROOT, load_classes

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def read_source_names(src: Path):
    ymls = list(src.glob("data.yaml")) + list(src.glob("*.yaml"))
    if not ymls:
        raise SystemExit("data.yaml tidak ditemukan di folder sumber. Unduh dengan format 'YOLOv8' / 'YOLO'.")
    text = ymls[0].read_text(encoding="utf-8")
    m = re.search(r"^names:\s*(\[.*?\])\s*$", text, re.M | re.S)
    if m:
        return [str(x) for x in ast.literal_eval(m.group(1))]
    m = re.search(r"^names:\s*\n((?:\s+.*\n?)+)", text, re.M)
    if m:
        names = {}
        for line in m.group(1).splitlines():
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                names[int(k)] = v.strip().strip("'\"")
            elif line.startswith("-"):
                names[len(names)] = line[1:].strip().strip("'\"")
        return [names[i] for i in sorted(names)]
    raise SystemExit("Tidak bisa membaca 'names' dari data.yaml.")


def source_key(p: Path):
    """Nama asal tanpa akhiran augmentasi Roboflow (.rf.<hash>), agar salinan augmentasi satu kelompok."""
    return re.split(r"\.rf\.", p.stem)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder hasil unduhan (berisi data.yaml, train/, valid/, ...)")
    ap.add_argument("--lihat-kelas", action="store_true", help="hanya tampilkan nama kelas sumber lalu keluar")
    ap.add_argument("--tag", default="pub", help="awalan unik untuk dataset ini, mis. vic1 (tanpa '_' dan spasi)")
    ap.add_argument("--map", action="append", default=[], metavar="SUMBER:TARGET",
                    help="pemetaan kelas, boleh diulang. TARGET harus ada di classes.txt")
    ap.add_argument("--semua-ke", "--map-semua", dest="semua_ke", default="", metavar="TARGET",
                    help="petakan SEMUA kelas sumber ke satu kelas target (mis. semua pose orang -> korban)")
    ap.add_argument("--n-sesi", type=int, default=10, help="jumlah 'sesi' hasil pemotongan data")
    ap.add_argument("--maks-per-kelas", type=int, default=0,
                    help="batasi jumlah kotak (instance) per kelas, mis. 50. 0 = ambil semua")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default="", help="URL halaman dataset (untuk provenance)")
    ap.add_argument("--lisensi", default="", help="lisensi dataset, mis. CC BY 4.0")
    a = ap.parse_args()

    src = Path(a.src)
    src_names = read_source_names(src)
    print("Kelas di dataset sumber:", dict(enumerate(src_names)))
    if a.lihat_kelas:
        return
    if "_" in a.tag or " " in a.tag:
        raise SystemExit("--tag tidak boleh mengandung '_' atau spasi.")
    if not a.map and not a.semua_ke:
        raise SystemExit("Isi --map (mis. --map victim:korban) atau --semua-ke korban")

    targets = load_classes()
    id_map = {}
    if a.semua_ke:
        if a.semua_ke not in targets:
            raise SystemExit(f"Kelas target '{a.semua_ke}' tidak ada di classes.txt: {targets}")
        id_map = {i: targets.index(a.semua_ke) for i in range(len(src_names))}
    for m in a.map:
        s, t = m.split(":", 1)
        if s not in src_names:
            raise SystemExit(f"Kelas sumber '{s}' tidak ada. Pilihan: {src_names}")
        if t not in targets:
            raise SystemExit(f"Kelas target '{t}' tidak ada di classes.txt. Isi classes.txt dulu: {targets[:5]}...")
        id_map[src_names.index(s)] = targets.index(t)

    # kumpulkan semua gambar dari train/valid/test (atau folder lain yang punya images/)
    items = []
    for imgdir in sorted(src.glob("*/images")):
        for p in sorted(imgdir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append((p, imgdir.parent / "labels" / (p.stem + ".txt")))
    if not items:
        raise SystemExit("Tidak ada gambar ditemukan. Struktur harus <src>/train/images, <src>/valid/images, dst.")

    # baca label tiap gambar, petakan kelas
    cand = []                                   # (key, gambar, baris_label_baru, had)
    for p, l in items:
        lines_out, had = [], False
        if l.exists():
            for line in l.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                had = True
                cid = int(float(parts[0]))
                if cid in id_map:
                    lines_out.append(f"{id_map[cid]} " + " ".join(parts[1:5]))
        cand.append((source_key(p), p, lines_out, had))

    skipped = sum(1 for c in cand if c[3] and not c[2])    # ada objek, tapi bukan kelas yang dipakai
    cand = [c for c in cand if not (c[3] and not c[2])]

    if a.maks_per_kelas > 0:
        import random
        rnd = random.Random(a.seed)
        pool = [c for c in cand if c[2]]                    # hanya gambar yang punya objek
        rnd.shuffle(pool)
        count, chosen = {}, []
        for c in pool:
            cls = {int(x.split()[0]) for x in c[2]}
            if any(count.get(k, 0) < a.maks_per_kelas for k in cls):
                chosen.append(c)
                for x in c[2]:
                    k = int(x.split()[0])
                    count[k] = count.get(k, 0) + 1
            if all(count.get(k, 0) >= a.maks_per_kelas for k in set(id_map.values())):
                break
        cand = chosen
        print("Instance terpilih per kelas:", {targets[k]: v for k, v in sorted(count.items())})
        for k in set(id_map.values()):
            if count.get(k, 0) < a.maks_per_kelas:
                print(f"[PERINGATAN] kelas '{targets[k]}' hanya {count.get(k,0)} instance di dataset ini "
                      f"(< {a.maks_per_kelas}). Tambah dataset lain.")

    if not cand:
        raise SystemExit("Tidak ada gambar yang cocok dengan --map.")

    # kelompokkan per gambar sumber, urut nama -> potong menjadi n sesi
    groups = {}
    for key, p, lines_out, had in cand:
        groups.setdefault(key, []).append((p, lines_out))
    keys = sorted(groups)
    n = max(1, min(a.n_sesi, len(keys)))
    chunk_of = {k: min(i * n // len(keys), n - 1) for i, k in enumerate(keys)}

    raw_img, raw_lbl = ROOT / "dataset_raw" / "images", ROOT / "dataset_raw" / "labels"
    raw_img.mkdir(parents=True, exist_ok=True)
    raw_lbl.mkdir(parents=True, exist_ok=True)
    meta = ROOT / "dataset_raw" / "metadata.csv"
    prov = ROOT / "dataset_raw" / "sumber_publik.csv"
    new_meta, new_prov = not meta.exists(), not prov.exists()

    kept = inst = 0
    counter = {}
    with open(meta, "a", newline="", encoding="utf-8") as fm, open(prov, "a", newline="", encoding="utf-8") as fp:
        wm, wp = csv.writer(fm), csv.writer(fp)
        if new_meta:
            wm.writerow(["nama_file", "sesi", "tanggal", "kondisi_cahaya", "jarak"])
        if new_prov:
            wp.writerow(["nama_file", "sumber_dataset", "url", "lisensi", "nama_asli"])
        for k in keys:
            sesi = f"{a.tag}c{chunk_of[k]:02d}"
            for p, lines_out in groups[k]:
                counter[sesi] = counter.get(sesi, 0) + 1
                name = f"{sesi}_pub_{counter[sesi]:04d}"
                shutil.copy2(p, raw_img / (name + p.suffix.lower()))
                if lines_out:
                    (raw_lbl / (name + ".txt")).write_text("\n".join(lines_out) + "\n")
                inst += len(lines_out)
                kept += 1
                wm.writerow([name + p.suffix.lower(), sesi, date.today().isoformat(), "publik", "publik"])
                wp.writerow([name + p.suffix.lower(), src.name, a.url, a.lisensi, p.name])

    print(f"Selesai: {kept} gambar diimpor ({inst} kotak), {skipped} dilewati (tidak ada kelas yang dipetakan).")
    print(f"Jumlah sesi dibuat: {len(counter)}  | catatan asal data: dataset_raw/sumber_publik.csv")
    print("Berikutnya: python split.py --val-ratio 0.2  lalu  python check_dataset.py --target 50")


if __name__ == "__main__":
    main()
