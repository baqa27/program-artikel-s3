"""Menggabungkan label dua penilai menjadi label final.

Pakai:
  python gabung_label.py penilai1_utama.xlsx+penilai1_tambahan.xlsx penilai2_utama.xlsx+penilai2_tambahan.xlsx \\
         lembar_diskusi.xlsx label_final.csv  (lembar diskusi boleh .xlsx atau .csv)

Aturan:
1. Pasangan yang labelnya sama pada dua penilai: label itu menjadi label final (sumber "sepakat").
2. Pasangan yang labelnya beda dibahas kedua penilai berdasarkan pedoman. Hasilnya ditulis di kolom
   label_sepakat pada lembar_diskusi.csv (sumber "diskusi").
3. Bila diskusi buntu, pasangan itu diserahkan ke pembimbing dan hasilnya ditulis di kolom label_pembimbing.
   Kolom ini menang atas label_sepakat (sumber "pembimbing").
Skrip berhenti bila masih ada pasangan beda yang belum punya label final sah (0, 1, atau 2).
"""
import sys

import pandas as pd

SAH = {"0", "1", "2"}


def bersih(seri):
    return seri.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def baca(path):
    # beberapa lembar milik satu penilai digabung dengan tanda +, misalnya utama.xlsx+tambahan.xlsx
    d = pd.concat([pd.read_excel(p, sheet_name="Penilaian", dtype=str) for p in path.split("+")], ignore_index=True)
    d["label"] = bersih(d["label"])
    return d[["no", "label"]]


p1, p2, diskusi, keluar = sys.argv[1:5]
m = baca(p1).merge(baca(p2), on="no", suffixes=("_1", "_2"))
if diskusi.lower().endswith(".xlsx"):
    t = pd.read_excel(diskusi, sheet_name="Diskusi", dtype=str).rename(
        columns={"Kode": "no", "Label Sepakat": "label_sepakat", "Label Pembimbing": "label_pembimbing"})
else:
    t = pd.read_csv(diskusi, dtype=str)
t = t[["no", "label_sepakat", "label_pembimbing"]]
for kolom in ("label_sepakat", "label_pembimbing"):
    t[kolom] = bersih(t[kolom])
m = m.merge(t, on="no", how="left")
sepakat = m.label_1 == m.label_2
pembimbing = m.label_pembimbing.fillna("").isin(SAH)
diskusi_ok = m.label_sepakat.fillna("").isin(SAH)
m["label"] = m.label_1.where(sepakat, m.label_sepakat.where(~pembimbing, m.label_pembimbing))
m["sumber_label"] = "sepakat"
m.loc[~sepakat & diskusi_ok, "sumber_label"] = "diskusi"
m.loc[~sepakat & pembimbing, "sumber_label"] = "pembimbing"
kosong = m[~m.label.isin(SAH)]
if len(kosong):
    sys.exit(f"{len(kosong)} pasangan beda belum punya label final yang sah (cek kolom label_sepakat dan label_pembimbing). "
             f"Contoh: {kosong.no.head().tolist()}")
m[["no", "label", "sumber_label"]].to_csv(keluar, index=False)
print(f"Label final: {len(m)} pasangan")
print(m.sumber_label.value_counts().to_string())
print(m.label.value_counts().sort_index().to_string())
