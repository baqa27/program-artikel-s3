"""Memilih 100 kueri dari judul Ekapta dalam satu prosedur (30 tuning, 70 uji).

Aturan:
1. Populasi: 354 judul Ekapta. Judul asal 80 pasangan buatan (20 judul) tidak dipakai sebagai kueri.
2. Satu judul per kode pengaju: bila seorang pengaju punya dua judul, satu dipilih acak.
3. Kuota 100 kueri dibagi proporsional menurut strata tahun x program studi pada 354 judul
   (metode sisa terbesar), lalu judul dalam tiap strata diambil acak.
4. Kuota 30 kueri tuning dibagi proporsional dari kuota strata dengan cara yang sama;
   sisanya menjadi kueri uji.
Semua langkah acak memakai satu generator dengan seed 42, jadi hasilnya bisa diulang persis.

Pakai: python pilih_kueri.py dataset_ekapta_354.csv buatan_v2_kunci.csv kueri_100.csv
"""
import sys

import numpy as np
import pandas as pd

N_KUERI, N_TUNING, SEED = 100, 30, 42


def bagi_proporsional(bobot: pd.Series, total: int) -> pd.Series:
    kuota = bobot / bobot.sum() * total
    hasil = np.floor(kuota).astype(int)
    for s in (kuota - hasil).sort_values(ascending=False, kind="stable").index[: total - hasil.sum()]:
        hasil[s] += 1
    return hasil


ekapta, buatan, keluaran = sys.argv[1:4]
A = pd.read_csv(ekapta, dtype=str)
A["tahun"] = A.tahun.str.slice(0, 4)
A["strata"] = A.tahun + " | " + A.prodi_standar
sumber_buatan = set(pd.read_csv(buatan, dtype=str).doc_id_sumber)
rng = np.random.default_rng(SEED)

target = bagi_proporsional(A.strata.value_counts().sort_index(), N_KUERI)
target_tuning = bagi_proporsional(target, N_TUNING)

kandidat = A[~A.doc_id.isin(sumber_buatan)].sort_values("doc_id").reset_index(drop=True)
kandidat = kandidat.iloc[rng.permutation(len(kandidat))].drop_duplicates("pengaju_kode").sort_values("doc_id")

pilih = []
for s in target.index:
    k = kandidat[kandidat.strata == s]
    n = int(target[s])
    if len(k) < n:
        sys.exit(f"Strata {s} hanya punya {len(k)} kandidat untuk kuota {n}.")
    ambil = k.iloc[rng.permutation(len(k))[:n]].copy()
    ambil["set"] = ["tuning"] * int(target_tuning[s]) + ["uji"] * (n - int(target_tuning[s]))
    pilih.append(ambil)
Q = pd.concat(pilih).sort_values("doc_id")
Q[["doc_id", "judul_bersih", "tokens_std", "tahun", "prodi_standar", "pengaju_kode", "set"]].to_csv(keluaran, index=False)

print(pd.DataFrame(dict(populasi=A.strata.value_counts().sort_index(), kueri=target, tuning=target_tuning,
                        uji=target - target_tuning)).to_string())
print(f"\nTotal kueri: {len(Q)} {Q.set.value_counts().to_dict()} | pengaju unik: {Q.pengaju_kode.nunique()}")
print(f"Judul asal pasangan buatan yang dikecualikan: {len(sumber_buatan)}")
