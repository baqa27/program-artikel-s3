"""Hitung Cohen's Kappa dua penilai (label bertingkat 2/1/0) dan siapkan lembar diskusi untuk pasangan yang labelnya beda.

Pakai:
  python hitung_kappa.py penilai1_utama.xlsx+penilai1_tambahan.xlsx penilai2_utama.xlsx+penilai2_tambahan.xlsx [kunci_penilaian_lengkap.csv] [--keluaran lembar_diskusi.csv]

Melaporkan dua nilai kappa (Cohen, 1960):
  - kappa tiga kategori (2, 1, 0)
  - kappa biner: mirip (2) melawan bukan mirip (1 atau 0), sesuai label positif untuk precision, recall, dan F1
Tiap nilai disertai proporsi sepakat (po) dan interval bootstrap 95% (2.000 kali, seed 42).
Bila berkas kunci diberikan, kappa dihitung juga per sumber pasangan (pool_3algoritma, buatan_v2).
Berkas keluaran: lembar_diskusi.csv (atau nama dari --keluaran). Isinya pasangan yang labelnya beda, dengan label kedua penilai,
kolom label_sepakat (diisi setelah dua penilai berdiskusi), dan kolom label_pembimbing (diisi hanya bila diskusi buntu).
Kappa dihitung dari label sebelum diskusi. Pakai --keluaran untuk lembar pilot supaya berkas utama tidak tertimpa.
Skrip berhenti bila ada baris kosong atau label di luar 2/1/0.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SAH = {"0", "1", "2"}

def baca(path):
    # beberapa lembar milik satu penilai digabung dengan tanda +, misalnya utama.xlsx+tambahan.xlsx
    d = pd.concat([pd.read_excel(p, sheet_name="Penilaian", dtype=str) for p in path.split("+")], ignore_index=True)
    d["label"] = d["label"].str.strip().str.replace(r"\.0$", "", regex=True)
    return d

def kappa_ik(y1, y2, seed=42, n=2000):
    k = cohen_kappa_score(y1, y2)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        i = rng.integers(0, len(y1), len(y1))
        if len(set(y1[i]) | set(y2[i])) > 1:
            bs.append(cohen_kappa_score(y1[i], y2[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan"))
    return k, float(np.mean(y1 == y2)), lo, hi

args = sys.argv[1:]
keluaran = "lembar_diskusi.csv"
if "--keluaran" in args:
    i = args.index("--keluaran"); keluaran = args[i + 1]; del args[i:i + 2]
f1, f2 = args[0], args[1]
a, b = baca(f1), baca(f2)
for nama, d in [(f1, a), (f2, b)]:
    kosong = int(d["label"].isna().sum()); salah = int((~d["label"].dropna().isin(SAH)).sum())
    if kosong or salah:
        sys.exit(f"{nama}: {kosong} baris kosong dan {salah} label tidak valid. Lengkapi dulu, kappa belum bisa dihitung.")
m = a[["no", "judul_1", "judul_2", "label", "yakin"]].merge(b[["no", "label", "yakin"]], on="no", suffixes=("_1", "_2"))
if len(m) != len(a) or len(m) != len(b):
    sys.exit("Nomor pasangan di dua berkas tidak sama. Pastikan keduanya dari set yang sama.")
y1, y2 = m.label_1.values, m.label_2.values
b1, b2 = (y1 == "2").astype(int), (y2 == "2").astype(int)
print(f"Jumlah pasangan: {len(m)}")
for nama, (u, v) in [("Tiga kategori (2/1/0)", (y1, y2)), ("Biner (2 vs 1/0)", (b1, b2))]:
    k, po, lo, hi = kappa_ik(u, v)
    print(f"{nama}: po={po:.4f}, kappa={k:.4f} (IK 95%: {lo:.4f} sampai {hi:.4f})")
print(pd.crosstab(m.label_1, m.label_2, rownames=["penilai1"], colnames=["penilai2"]).to_string())
print("Baris ragu: penilai1 =", int((m.yakin_1 == "ragu").sum()), "| penilai2 =", int((m.yakin_2 == "ragu").sum()))
if len(args) > 2:
    kunci = pd.read_csv(args[2], dtype=str)[["no", "sumber_pasangan"]]
    mm = m.merge(kunci, on="no")
    for s, g in mm.groupby("sumber_pasangan"):
        u, v = g.label_1.values, g.label_2.values
        k = cohen_kappa_score(u, v) if len(set(u) | set(v)) > 1 else float("nan")
        print(f"  {s}: n={len(g)}, po={np.mean(u == v):.4f}, kappa 3 kategori={k:.4f}")
beda = m[m.label_1 != m.label_2][["no", "judul_1", "judul_2", "label_1", "label_2"]].rename(
    columns={"label_1": "label_penilai1", "label_2": "label_penilai2"})
beda["label_sepakat"] = ""
beda["label_pembimbing"] = ""
beda.to_csv(keluaran, index=False)
print(f"Pasangan beda label: {len(beda)} ({len(beda) / len(m):.1%}) -> {keluaran}")
