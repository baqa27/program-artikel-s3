"""Bukti Pendahuluan: pasangan judul Ekapta dari pengaju berbeda yang mirip, dan berapa yang ditandai SQL LIKE.

Pakai: python cek_like_ekapta.py ../04_data/korpus/dataset_ekapta_354.csv
Jaccard dihitung pada tokens_std; SQL LIKE (kemiripan/sql_like.py) pada judul_asli, dua arah.
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kemiripan.sql_like import sql_like_cocok

E = pd.read_csv(sys.argv[1], dtype=str)
T = [set(t.split()) for t in E.tokens_std]
rows = []
for i, j in itertools.combinations(range(len(E)), 2):
    if E.pengaju_kode[i] == E.pengaju_kode[j]:
        continue
    s = len(T[i] & T[j]) / len(T[i] | T[j])
    if s >= 0.5:
        like = sql_like_cocok(E.judul_asli[i], E.judul_asli[j]) or sql_like_cocok(E.judul_asli[j], E.judul_asli[i])
        rows.append((E.doc_id[i], E.doc_id[j], round(s, 3), like, E.judul_asli[i], E.judul_asli[j]))
d = pd.DataFrame(rows, columns=["doc_id_a", "doc_id_b", "jaccard", "ditandai_like", "judul_a", "judul_b"])
for t in (0.5, 0.6, 0.7, 0.8):
    x = d[d.jaccard >= t]
    print(f"Jaccard >= {t}: {len(x)} pasangan, ditandai SQL LIKE: {int(x.ditandai_like.sum())}")
print("\nPasangan dengan Jaccard >= 0,8:")
for r in d[d.jaccard >= 0.8].sort_values("jaccard", ascending=False).itertuples():
    print(f"  {r.doc_id_a} - {r.doc_id_b} ({r.jaccard}): {r.judul_a} || {r.judul_b}")
