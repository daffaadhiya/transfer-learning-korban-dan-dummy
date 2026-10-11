"""Gabungkan hasil 3 mode: tabel (results/ringkasan.md) dan grafik mAP per epoch."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

MODES = ["feature", "partial", "scratch"]
res = ROOT / "results"
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
rows = []
for m in MODES:
    log, summ = res / f"{m}_log.csv", res / f"{m}_summary.json"
    if not log.exists():
        continue
    d = pd.read_csv(log)
    ax[0].plot(d["epoch"], d["map50"], label=m)
    ax[1].plot(d["epoch"], d["train_loss"], label=m)
    s = json.loads(summ.read_text())
    rows.append({"Mode": m, "mAP50 terbaik": f"{s['best_map50']:.3f}",
                 "mAP50-95 terbaik": f"{s['best_map50_95']:.3f}",
                 "Waktu latih (s)": s["waktu_latih_s"],
                 f"Epoch mAP50>={s['target_map50']}": s["epoch_map50_ke_target"] or "tidak tercapai",
                 "Param dilatih (M)": f"{s['param_dilatih']/1e6:.2f}"})
ax[0].set(title="mAP50 validasi per epoch", xlabel="epoch", ylabel="mAP50")
ax[1].set(title="Loss training per epoch", xlabel="epoch", ylabel="loss")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(res / "kurva_3_mode.png", dpi=150)
if rows:
    (res / "ringkasan.md").write_text(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    print((res / "ringkasan.md").read_text())
