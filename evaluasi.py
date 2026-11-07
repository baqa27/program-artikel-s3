"""Eksperimen lengkap artikel: penyetelan, evaluasi set uji, uji statistik, dan bahan analisis kesalahan.

Pakai (setelah label final ada):
  python evaluasi.py korpus_master_v4.csv kueri_100.csv kunci_penilaian_lengkap.csv label_final.csv keluaran/ \
         --karya-sama karya_sama_ekapta_repo.csv [--sql-like]

Isi keluaran/:
  tabel6_parameter.csv         parameter terpilih per algoritma dan skenario (set tuning, F1 tertinggi)
  tabel7_kinerja.csv           precision, recall, F1 [IK 95%], nDCG@5 [IK 95%] pada set uji
  tabel8_signifikansi.csv      selisih dan p terkoreksi Holm untuk tiga pasangan algoritma
  tabel5_sebaran_label.csv     sebaran label final per set
  sensitivitas_ambang.csv      F1 set tuning untuk setiap ambang (dan k1, b untuk BM25)
  ndcg_per_kueri.csv           nDCG@5 setiap kueri uji (dasar uji randomisasi)
  kesalahan_uji_skenarioA.csv  pasangan uji yang salah diprediksi minimal satu algoritma; isi kolom kategori manual
  belum_dinilai_top5.csv       pasangan kueri uji di lima teratas yang tidak ikut pooling (biasanya dari BM25 dengan
                               k1 dan b hasil penyetelan); dinilai dua penilai lalu dimasukkan lewat --label-tambahan
  ringkasan.txt                semua angka di atas dalam format desimal koma, siap dipindah ke artikel

Aturan yang dipakai (sama dengan bagian Metode artikel):
  - judul Ekapta dan salinannya sendiri di repositori (karya yang sama, kolom tinjauan = "ya") diperlakukan seperti
    dua judul dari pengaju yang sama: pasangannya dikeluarkan dari evaluasi dan salinannya tidak ikut peringkat
  - kelas mirip = label 2; algoritma memprediksi mirip bila skor >= ambang
  - ambang 0,05-0,95 (langkah 0,05); BM25: k1 0,5-2,0 (0,25), b 0-1 (0,25)
  - konfigurasi dipilih dari F1 set tuning, lalu dipakai apa adanya pada set uji
  - nDCG@5 dihitung per kueri; pasangan yang tidak dinilai dianggap tidak relevan (rel = 0);
    kueri tanpa judul berlabel 1 atau 2 dikeluarkan dan jumlahnya dilaporkan; proporsi lima teratas yang sudah dinilai
    dilaporkan sebagai cakupan_dinilai_top5
  - IK 95%: bootstrap atas kueri (10.000 kali, seed 42). Untuk precision/recall/F1 dipakai bootstrap klaster:
    yang diambil ulang adalah kueri beserta semua pasangannya (pasangan buatan dikelompokkan per judul asal),
    karena pasangan dari kueri yang sama tidak saling bebas
  - uji randomisasi berpasangan (pembalikan tanda per kueri, 10.000 kali) untuk nDCG@5;
    bootstrap klaster berpasangan (per kueri) untuk F1; koreksi Holm untuk tiga pasangan per metrik per skenario
"""
from __future__ import annotations

import argparse
import itertools
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kemiripan import ALGORITMA, top_k
from kemiripan._umum import bersihkan
from kemiripan.bm25 import bm25_simetris
from kemiripan.bm25 import skor_judul_baru as bm25_baru
from kemiripan.jaccard import jaccard
from kemiripan.jaccard import skor_judul_baru as jaccard_baru
from kemiripan.praproses import praproses
from kemiripan.tfidf_cosine import skor_judul_baru as tfidf_baru
from kemiripan.tfidf_cosine import tfidf_cosine

AMBANG = np.round(np.arange(0.05, 0.951, 0.05), 2)
K1 = np.round(np.arange(0.5, 2.001, 0.25), 2)
B = np.round(np.arange(0.0, 1.001, 0.25), 2)
NAMA = {"jaccard": "Jaccard", "tfidf": "TF-IDF Cosine", "bm25": "BM25"}

ap = argparse.ArgumentParser()
ap.add_argument("korpus"); ap.add_argument("kueri"); ap.add_argument("kunci"); ap.add_argument("label"); ap.add_argument("keluaran")
ap.add_argument("--iterasi", type=int, default=10_000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--karya-sama", help="CSV karya_sama_ekapta_repo.csv; pasangan berstatus ya dikeluarkan")
ap.add_argument("--label-tambahan", help="CSV kolom kueri, doc_id, label: label final untuk isi belum_dinilai_top5.csv")
ap.add_argument("--sql-like", action="store_true", help="tambahkan pembanding dasar Check Plagiarism Ekapta (SQL LIKE)")
arg = ap.parse_args()
os.makedirs(arg.keluaran, exist_ok=True)
t0 = time.time()

K = pd.read_csv(arg.korpus, dtype=str)
Q = pd.read_csv(arg.kueri, dtype=str)
L = pd.read_csv(arg.kunci, dtype=str).merge(pd.read_csv(arg.label, dtype=str)[["no", "label"]], on="no", how="left")
if L.label.isna().any():
    sys.exit(f"{int(L.label.isna().sum())} pasangan belum punya label final.")
L["label"] = L.label.astype(int)
KS, ks_kueri = set(), {}
if arg.karya_sama:
    ks = pd.read_csv(arg.karya_sama, dtype=str)
    ks = ks[ks.karya_sama_tinjauan_awal_AI.eq("ya")]
    for a_, b_ in zip(ks.doc_id_ekapta, ks.doc_id_repo):
        KS.add(frozenset((a_, b_))); ks_kueri.setdefault(a_, set()).add(b_)
    keluar = np.array([frozenset((a_, b_)) in KS for a_, b_ in zip(L.doc_id_1, L.doc_id_2)])
    L = L[~keluar].reset_index(drop=True)
    print(f"Pasangan karya sama yang dikeluarkan: {int(keluar.sum())}")
else:
    print("PERINGATAN: --karya-sama tidak diberikan; salinan karya sendiri di repositori ikut dihitung.")
n_karya_sama = len(KS)
L["buatan"] = L.sumber_pasangan.eq("buatan_v2")
L["di_A"] = L.di_skenario_A.astype(str).eq("True")
qset = dict(zip(Q.doc_id, Q.set))

# label pasangan alami untuk peringkat: {frozenset(doc1, doc2): label}
alami = L[~L.buatan]
label_alami = {frozenset((a, b)): l for a, b, l in zip(alami.doc_id_1, alami.doc_id_2, alami.label)}
dinilai = {}   # kueri -> [(judul lain, label)] untuk menghitung IDCG
for a, b, l in zip(alami.doc_id_1, alami.doc_id_2, alami.label):
    dinilai.setdefault(a, []).append((b, l)); dinilai.setdefault(b, []).append((a, l))
if arg.label_tambahan:
    LT = pd.read_csv(arg.label_tambahan, dtype=str)
    for a, b, l in zip(LT.kueri, LT.doc_id, LT.label.astype(int)):
        if frozenset((a, b)) not in label_alami:
            label_alami[frozenset((a, b))] = l
            dinilai.setdefault(a, []).append((b, l)); dinilai.setdefault(b, []).append((a, l))
    print(f"Label tambahan untuk nDCG: {len(LT)} pasangan")

KORPUS = {"A": K[K.skenario_A == "True"].reset_index(drop=True), "B": K.reset_index(drop=True)}


def f1_pr(y, p):
    tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def ndcg5(gains, ideal):
    disc = 1 / np.log2(np.arange(2, 7))
    dcg = float(np.sum(np.asarray(gains[:5]) * disc[: len(gains[:5])]))
    ideal = sorted(ideal, reverse=True)[:5]
    idcg = float(np.sum(np.asarray(ideal) * disc[: len(ideal)]))
    return dcg / idcg if idcg > 0 else np.nan


hasil_tuning, hasil_uji, per_kueri, sens, belum = [], [], [], [], []
skor_uji = {}   # (skenario, alg) -> (pairs dataframe, skor, pred)

for sk, df in KORPUS.items():
    tok = bersihkan(df.tokens_std.tolist())
    ids = df.doc_id.values
    pos = {d: i for i, d in enumerate(ids)}
    peng = df.pengaju_kode.fillna("").values
    P = L[L.buatan | (L.di_A if sk == "A" else True)].copy().reset_index(drop=True)
    nat = ~P.buatan.values
    i1 = np.array([pos.get(d, -1) for d in P.doc_id_1]); i2 = np.array([pos.get(d, -1) for d in P.doc_id_2])
    buatan_pas = [(praproses(j2), pos[d1]) for d1, j2 in zip(P.doc_id_1[~nat], P.judul_2[~nat])]
    y = P.label.values == 2
    tun = P.set_klasifikasi.values == "tuning"

    def skor_pasangan(M, baru):
        s = np.empty(len(P))
        s[nat] = M[i1[nat], i2[nat]]
        s[~nat] = baru
        return s

    kandidat = {}
    J = jaccard(tok); kandidat["jaccard"] = [((), J, skor_pasangan(J, jaccard_baru(tok, buatan_pas)))]
    T = tfidf_cosine(tok); kandidat["tfidf"] = [((), T, skor_pasangan(T, tfidf_baru(tok, buatan_pas)))]
    kandidat["bm25"] = []
    for k1, b in itertools.product(K1, B):
        M = bm25_simetris(tok, k1, b)
        kandidat["bm25"].append(((k1, b), None, skor_pasangan(M, bm25_baru(tok, buatan_pas, k1, b))))
        del M   # matriks 5.229 x 5.229 tidak disimpan semua agar memori cukup; dihitung ulang untuk konfigurasi terpilih

    # 1. penyetelan pada set tuning
    for alg in ALGORITMA:
        terbaik = None
        for par, M, s in kandidat[alg]:
            for a in AMBANG:
                pr, rc, f1 = f1_pr(y[tun], s[tun] >= a)
                sens.append(dict(skenario=sk, algoritma=alg, k1=par[0] if par else "", b=par[1] if par else "",
                                 ambang=a, precision=pr, recall=rc, f1=f1))
                if terbaik is None or f1 > terbaik[0] + 1e-12:
                    terbaik = (f1, par, a, M, s)
        f1t, par, a, M, s = terbaik
        if M is None:
            M = bm25_simetris(tok, par[0], par[1])
        hasil_tuning.append(dict(skenario=sk, algoritma=alg, k1=par[0] if par else "", b=par[1] if par else "",
                                 ambang=a, f1_tuning=f1t))
        # 2. evaluasi pada set uji
        uji = ~tun
        pred = s >= a
        pr, rc, f1 = f1_pr(y[uji], pred[uji])
        skor_uji[(sk, alg)] = (P[uji].reset_index(drop=True), s[uji], pred[uji], y[uji])
        for q in Q.doc_id[Q.set == "uji"]:
            i = pos[q]
            ex = ((peng == peng[i]) & (peng != "")) | np.isin(ids, list(ks_kueri.get(q, ())))
            urut = top_k(M, i, 5, ex)
            gains = [label_alami.get(frozenset((q, ids[j])), 0) for j in urut]
            sudah = [frozenset((q, ids[j])) in label_alami for j in urut]
            for j, ok in zip(urut, sudah):
                if not ok:
                    belum.append(dict(skenario=sk, algoritma=alg, k1=par[0] if par else "", b=par[1] if par else "",
                                      kueri=q, doc_id=ids[j], judul_kueri=df.judul_bersih[i], judul_lain=df.judul_bersih[j]))
            ideal = [l for d, l in dinilai.get(q, []) if d in pos and not (peng[pos[d]] == peng[i] and peng[i] != "")
                     and d not in ks_kueri.get(q, ())]
            per_kueri.append(dict(skenario=sk, algoritma=alg, kueri=q, ndcg5=ndcg5(gains, ideal),
                                  dinilai_top5=int(sum(sudah)), n_top5=len(urut)))
        hasil_uji.append(dict(skenario=sk, algoritma=alg, precision=pr, recall=rc, f1=f1,
                              n_pasangan_uji=int(uji.sum()), n_mirip_uji=int(y[uji].sum())))

    if arg.sql_like and sk == "A":
        from kemiripan.sql_like import sql_like_cocok
        asli = dict(zip(K.doc_id, K.judul_asli))
        Pu = P[~tun]
        ja = [asli.get(d, d) for d in Pu.doc_id_1]
        jb = [j2 if bt else asli.get(d2, d2) for d2, j2, bt in zip(Pu.doc_id_2, Pu.judul_2, Pu.buatan)]
        pred = np.array([sql_like_cocok(a_, b_) or sql_like_cocok(b_, a_) for a_, b_ in zip(ja, jb)])
        pr, rc, f1 = f1_pr(Pu.label.values == 2, pred)
        hasil_uji.append(dict(skenario=sk, algoritma="sql_like", precision=pr, recall=rc, f1=f1,
                              n_pasangan_uji=len(Pu), n_mirip_uji=int((Pu.label.values == 2).sum())))

U = pd.DataFrame(hasil_uji); T6 = pd.DataFrame(hasil_tuning); NK = pd.DataFrame(per_kueri)


def kueri_utama(teks):
    # pasangan antara dua kueri dimasukkan ke klaster kueri uji yang pertama
    q = str(teks).split("|")
    return next((x for x in q if qset.get(x) == "uji"), q[0])


def klaster(Pu):
    kunci = np.where(Pu.sumber_pasangan.eq("buatan_v2").values, "B:" + Pu.doc_id_1.astype(str).values,
                     [kueri_utama(t) for t in Pu.kueri_terkait])
    return [np.asarray(ix) for ix in pd.Series(np.arange(len(Pu))).groupby(kunci).apply(list)]


def ambil_ulang(grup, rng):
    return np.concatenate([grup[g] for g in rng.integers(0, len(grup), len(grup))])


rng = np.random.default_rng(arg.seed)

# 3. interval kepercayaan
for idx, r in U.iterrows():
    if r.algoritma == "sql_like":
        continue
    Pu, s, pred, y = skor_uji[(r.skenario, r.algoritma)]
    grup = klaster(Pu); f1s = []
    for _ in range(arg.iterasi):
        ii = ambil_ulang(grup, rng)
        f1s.append(f1_pr(y[ii], pred[ii])[2])
    U.loc[idx, "n_klaster_f1"] = len(grup)
    U.loc[idx, ["f1_ik_bawah", "f1_ik_atas"]] = np.percentile(f1s, [2.5, 97.5])
    v = NK[(NK.skenario == r.skenario) & (NK.algoritma == r.algoritma)].ndcg5.dropna().values
    U.loc[idx, "ndcg5"] = v.mean()
    U.loc[idx, "n_kueri_ndcg"] = len(v)
    nk = NK[(NK.skenario == r.skenario) & (NK.algoritma == r.algoritma)]
    U.loc[idx, "cakupan_dinilai_top5"] = nk.dinilai_top5.sum() / nk.n_top5.sum()
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(arg.iterasi)]
    U.loc[idx, ["ndcg5_ik_bawah", "ndcg5_ik_atas"]] = np.percentile(bs, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r_, i in enumerate(o):
        run = max(run, (m - r_) * p[i]); adj[i] = min(run, 1.0)
    return adj


# 4. uji signifikansi
sig = []
for sk in KORPUS:
    W = NK[NK.skenario == sk].pivot(index="kueri", columns="algoritma", values="ndcg5").dropna()
    baris = []
    for a, b in [("bm25", "tfidf"), ("bm25", "jaccard"), ("tfidf", "jaccard")]:
        d = (W[a] - W[b]).values; obs = d.mean()
        tanda = rng.choice([-1, 1], size=(arg.iterasi, len(d)))
        p_nd = (np.sum(np.abs((tanda * d).mean(axis=1)) >= abs(obs) - 1e-12) + 1) / (arg.iterasi + 1)
        Pa, _, pa, ya = skor_uji[(sk, a)]; _, _, pb, _ = skor_uji[(sk, b)]
        grup = klaster(Pa); diffs = []
        for _ in range(arg.iterasi):
            ii = ambil_ulang(grup, rng)
            diffs.append(f1_pr(ya[ii], pa[ii])[2] - f1_pr(ya[ii], pb[ii])[2])
        diffs = np.array(diffs)
        obs_f1 = f1_pr(ya, pa)[2] - f1_pr(ya, pb)[2]
        p_f1 = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
        baris.append(dict(skenario=sk, pasangan=f"{NAMA[a]} - {NAMA[b]}", selisih_ndcg5=obs, p_ndcg5=p_nd,
                          selisih_f1=obs_f1, p_f1=p_f1, n_kueri=len(d)))
    baris = pd.DataFrame(baris)
    baris["p_ndcg5_holm"] = holm(baris.p_ndcg5); baris["p_f1_holm"] = holm(baris.p_f1)
    sig.append(baris)
S8 = pd.concat(sig, ignore_index=True)

# 5. bahan analisis kesalahan (skenario A, set uji)
Pu = skor_uji[("A", "jaccard")][0].copy()
for alg in ALGORITMA:
    _, s, pred, y = skor_uji[("A", alg)]
    Pu[f"skor_{alg}"] = s; Pu[f"prediksi_{alg}"] = pred.astype(int)
salah = Pu[[any((Pu.label == 2).values[i] != Pu[f"prediksi_{a}"].values[i] for a in ALGORITMA) for i in range(len(Pu))]]
salah = salah[["no", "sumber_pasangan", "jenis_varian", "judul_1", "judul_2", "label"] +
              [c for a in ALGORITMA for c in (f"skor_{a}", f"prediksi_{a}")]].copy()
salah["kategori"] = ""
salah.to_csv(os.path.join(arg.keluaran, "kesalahan_uji_skenarioA.csv"), index=False)

# 6. sebaran label
T5 = L.groupby(["set_klasifikasi", "label"]).size().unstack(fill_value=0)

T6.to_csv(os.path.join(arg.keluaran, "tabel6_parameter.csv"), index=False)
U.to_csv(os.path.join(arg.keluaran, "tabel7_kinerja.csv"), index=False)
S8.to_csv(os.path.join(arg.keluaran, "tabel8_signifikansi.csv"), index=False)
T5.to_csv(os.path.join(arg.keluaran, "tabel5_sebaran_label.csv"))
pd.DataFrame(sens).to_csv(os.path.join(arg.keluaran, "sensitivitas_ambang.csv"), index=False)
NK.to_csv(os.path.join(arg.keluaran, "ndcg_per_kueri.csv"), index=False)
BL = pd.DataFrame(belum, columns=["skenario", "algoritma", "k1", "b", "kueri", "doc_id", "judul_kueri", "judul_lain"])
BL = BL.drop_duplicates(["kueri", "doc_id"]).assign(label_penilai1="", label_penilai2="", label="")
BL.to_csv(os.path.join(arg.keluaran, "belum_dinilai_top5.csv"), index=False)


def k(x, d=3):
    return f"{x:.{d}f}".replace(".", ",") if pd.notna(x) else "-"


def kp(p):
    return "< 0,001" if p < 0.001 else k(p)


with open(os.path.join(arg.keluaran, "ringkasan.txt"), "w", encoding="utf-8") as f:
    w = lambda s="": f.write(s + "\n")
    w("RINGKASAN HASIL (desimal koma, siap dipindah ke artikel)")
    w(f"Python {platform.python_version()}, seed {arg.seed}, iterasi {arg.iterasi}, waktu {time.time() - t0:.0f} detik")
    w("\nTabel 5. Sebaran label final (baris = set, kolom = label)"); w(T5.to_string())
    w("\nTabel 6. Parameter terpilih pada set tuning")
    for r in T6.itertuples():
        par = f"k1 = {k(r.k1, 2)}, b = {k(r.b, 2)}, " if r.algoritma == "bm25" else ""
        w(f"  Skenario {r.skenario} | {NAMA[r.algoritma]}: {par}ambang = {k(r.ambang, 2)} (F1 tuning {k(r.f1_tuning)})")
    w("\nTabel 7. Kinerja pada set uji")
    for r in U.itertuples():
        if r.algoritma == "sql_like":
            w(f"  Skenario {r.skenario} | SQL LIKE (Ekapta): P {k(r.precision)}, R {k(r.recall)}, F1 {k(r.f1)}")
            continue
        w(f"  Skenario {r.skenario} | {NAMA[r.algoritma]}: P {k(r.precision)}, R {k(r.recall)}, "
          f"F1 {k(r.f1)} [{k(r.f1_ik_bawah)}; {k(r.f1_ik_atas)}], nDCG@5 {k(r.ndcg5)} "
          f"[{k(r.ndcg5_ik_bawah)}; {k(r.ndcg5_ik_atas)}], kueri nDCG {int(r.n_kueri_ndcg)}, "
          f"lima teratas sudah dinilai {k(100 * r.cakupan_dinilai_top5, 1)}%, "
          f"pasangan uji {r.n_pasangan_uji} (mirip {r.n_mirip_uji})")
    w("\nTabel 8. Uji signifikansi (p terkoreksi Holm)")
    for r in S8.itertuples():
        w(f"  Skenario {r.skenario} | {r.pasangan}: selisih nDCG@5 {k(r.selisih_ndcg5)} (p {kp(r.p_ndcg5_holm)}), "
          f"selisih F1 {k(r.selisih_f1)} (p {kp(r.p_f1_holm)})")
    w(f"\nKarya sama Ekapta-repositori yang dikecualikan: {n_karya_sama} pasangan dalam daftar tinjauan"
      f" ({'dipakai' if arg.karya_sama else 'TIDAK dipakai'})")
    w(f"\nPasangan kueri uji di lima teratas yang belum dinilai: {len(BL)} (belum_dinilai_top5.csv). "
      "Bila lebih dari 0, minta kedua penilai melabelinya, isi kolom label, lalu jalankan ulang dengan --label-tambahan.")
    w(f"\nPasangan uji skenario A yang salah diprediksi minimal satu algoritma: {len(salah)} "
      f"(isi kolom kategori di kesalahan_uji_skenarioA.csv)")
print(open(os.path.join(arg.keluaran, "ringkasan.txt"), encoding="utf-8").read())
