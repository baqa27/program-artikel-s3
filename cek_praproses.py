"""Mencocokkan praproses.py dengan kolom judul_norm dan tokens_std korpus.

Pakai: python cek_praproses.py korpus_master_v4.csv [keluaran.txt]
"""
import collections, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kemiripan.praproses import normalisasi, praproses, token_dari_norm

K = pd.read_csv(sys.argv[1], dtype=str)
baris = []
def cetak(s):
    print(s); baris.append(s)
n1 = sum(normalisasi(b) == n for b, n in zip(K.judul_bersih, K.judul_norm))
cetak(f"Tahap normalisasi (judul_bersih -> judul_norm): {n1}/{len(K)} = {n1/len(K)*100:.2f}% judul identik")
for lang in ["Indonesia", "English", "Arab"]:
    sub = K[K.bahasa_final == lang]
    ok2 = sum(" ".join(token_dari_norm(n, lang)) == (t if isinstance(t, str) else "") for n, t in zip(sub.judul_norm, sub.tokens_std))
    ok3 = 0; selisih = collections.Counter()
    for b, t in zip(sub.judul_bersih, sub.tokens_std.fillna("")):
        v = praproses(b, lang)
        if v == t:
            ok3 += 1
        else:
            a, c = collections.Counter(v.split()), collections.Counter(t.split())
            for x in a - c: selisih["lebih:" + x] += 1
            for x in c - a: selisih["kurang:" + x] += 1
    cetak(f"{lang}: judul_norm -> tokens_std {ok2}/{len(sub)} ({ok2/len(sub)*100:.2f}%), "
          f"judul_bersih -> tokens_std {ok3}/{len(sub)} ({ok3/len(sub)*100:.2f}%)")
    cetak(f"   selisih token terbanyak: {selisih.most_common(12)}")
A = K[K.skenario_A == "True"]
ok = sum(praproses(b, l) == (t if isinstance(t, str) else "") for b, l, t in zip(A.judul_bersih, A.bahasa_final, A.tokens_std))
cetak(f"Judul Ekapta (354): judul_bersih -> tokens_std {ok}/{len(A)} ({ok/len(A)*100:.2f}%)")
if len(sys.argv) > 2:
    open(sys.argv[2], "w", encoding="utf-8").write("\n".join(baris) + "\n")
