"""Pooling top-5 tiga algoritma dan lembar penilaian buta untuk artikel (100 kueri).

Langkah:
1. Hitung skor Jaccard, TF-IDF Cosine, dan BM25 (k1 = 1,2; b = 0,75) pada skenario A (354 judul Ekapta)
   dan skenario B (5.229 judul).
2. Untuk setiap kueri, ambil 5 judul teratas tiap algoritma di tiap skenario. Judul itu sendiri dan
   judul dari kode pengaju yang sama dilewati.
3. Gabungkan pasangan tak berurut (A-B = B-A). Pasangan yang terkait kueri uji masuk set uji.
4. Tambahkan 80 pasangan buatan, acak urutan (seed 42), beri nomor P0001 dan seterusnya.
5. Tulis kunci_penilaian_artikel.csv (JANGAN diberikan ke penilai), lembar penilai 1 dan 2, dan lembar pilot.

Pakai: python siapkan_penilaian.py korpus_master_v4.csv kueri_100.csv buatan_v2_kunci.csv near_duplicate.csv keluaran_dir
"""
import os
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kemiripan import ALGORITMA, hitung_semua, top_k

master, kueri, buatan, neardup, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
K = pd.read_csv(master, dtype=str)
Q = pd.read_csv(kueri, dtype=str)
Bt = pd.read_csv(buatan, dtype=str)
ND = pd.read_csv(neardup)
nd = {tuple(sorted(x)): s for x, s in zip(zip(ND.doc_id_a, ND.doc_id_b), ND.skor)}
qset = dict(zip(Q.doc_id, Q.set))
info = K.set_index("doc_id")

rows = {}
for nama, df in (("A", K[K.skenario_A == "True"].reset_index(drop=True)), ("B", K.reset_index(drop=True))):
    S = hitung_semua(df.tokens_std.fillna("").tolist())
    ids = df.doc_id.values
    peng = df.pengaju_kode.fillna("").values
    pos = {d: i for i, d in enumerate(ids)}
    for q in Q.doc_id:
        i = pos[q]
        ex = (peng == peng[i]) & (peng != "")
        for alg in ALGORITMA:
            for j in top_k(S[alg], i, 5, ex):
                key = tuple(sorted((q, ids[j])))
                r = rows.setdefault(key, dict(kueri=set(), dipilih=set()))
                r["kueri"].add(q)
                r["dipilih"].add(f"{alg}@{nama}")

alami = pd.DataFrame([dict(sumber_pasangan="pool_3algoritma", doc_id_1=a, doc_id_2=b,
                           kueri_terkait="|".join(sorted(v["kueri"])),
                           set_kueri="|".join(sorted({qset[x] for x in v["kueri"]})),
                           dipilih_oleh="|".join(sorted(v["dipilih"])))
                      for (a, b), v in rows.items()])
alami["judul_1"] = alami.doc_id_1.map(info.judul_bersih)
alami["judul_2"] = alami.doc_id_2.map(info.judul_bersih)
alami["set_klasifikasi"] = alami.set_kueri.apply(lambda s: "uji" if "uji" in s else "tuning")
alami["lintas_set"] = alami.set_kueri.str.contains("[|]")
alami["di_skenario_A"] = alami.doc_id_1.map(info.skenario_A).eq("True") & alami.doc_id_2.map(info.skenario_A).eq("True")

def karya_sama(r):
    s = nd.get((r.doc_id_1, r.doc_id_2))
    a, b = r.doc_id_1, r.doc_id_2
    if s is None or s < 0.8 or a[:3] == b[:3]:
        return False
    e, rp = (a, b) if a.startswith("EKP") else (b, a)
    te = pd.to_numeric(info.tahun.get(e), errors="coerce")
    tr = pd.to_numeric(info.tahun.get(rp), errors="coerce")
    return bool(info.prodi_standar.get(e) == info.prodi_standar.get(rp) and pd.notna(te) and pd.notna(tr) and tr >= te)

alami["kandidat_karya_sama"] = alami.apply(karya_sama, axis=1)
alami["jenis_varian"] = ""
alami["rencana_label"] = ""
art = pd.DataFrame(dict(sumber_pasangan="buatan_v2", doc_id_1=Bt.doc_id_sumber, doc_id_2="(buatan)", kueri_terkait="",
                        set_kueri=Bt.set, dipilih_oleh="", judul_1=Bt.judul_a, judul_2=Bt.judul_b,
                        set_klasifikasi=Bt.set, lintas_set=False, di_skenario_A=True, kandidat_karya_sama=False,
                        jenis_varian=Bt.jenis_varian, rencana_label=Bt.rencana_label, pair_id_buatan=Bt.pair_id))
semua = pd.concat([alami, art], ignore_index=True)
rng = np.random.default_rng(42)
semua = semua.iloc[rng.permutation(len(semua))].reset_index(drop=True)
semua.insert(0, "no", [f"P{i:04d}" for i in range(1, len(semua) + 1)])
semua.to_csv(os.path.join(out, "kunci_penilaian_artikel.csv"), index=False)

# pilot: 40 pasangan alami dan 10 pasangan buatan (seed 7); pasangan pilot juga muncul di lembar utama
pil = pd.concat([semua[semua.sumber_pasangan == "pool_3algoritma"].sample(40, random_state=7),
                 semua[semua.sumber_pasangan == "buatan_v2"].sample(10, random_state=7)])

PETUNJUK = [
    "PETUNJUK PENILAIAN KEMIRIPAN PASANGAN JUDUL (artikel tiga algoritma)",
    "",
    "1. Isi lembar 'Identitas' dulu: inisial, peran, tanggal mulai.",
    "2. Nilai setiap baris secara mandiri. Jangan berdiskusi dengan penilai lain sebelum semua baris selesai.",
    "3. Kolom label hanya menerima angka 2, 1, atau 0:",
    "   2 = mirip          : masalah, objek, dan metode sama; beda hanya susunan kata, sinonim, singkatan, atau keterangan tambahan.",
    "                        Pengelola Ekapta akan meminta salah satu judul direvisi.",
    "   1 = mirip sebagian : masalah atau bidang aplikasi dan metode sama, tetapi objek atau lokasi studi kasus berbeda",
    "                        (contoh: sistem pakar Naive Bayes untuk penyakit tanaman teh dan tanaman jahe).",
    "   0 = tidak mirip    : masalah atau bidang berbeda, walaupun metode atau kerangka katanya sama",
    "                        (contoh: sistem pakar certainty factor untuk kerusakan komputer dan penyakit ayam).",
    "4. Kolom yakin: pilih 'ragu' bila keputusan sulit. Baris ragu tetap wajib diberi label.",
    "5. Kolom catatan boleh diisi alasan singkat, terutama untuk baris ragu.",
    "6. Abaikan huruf kapital, typo, dan urutan judul kiri/kanan. Nilai isinya.",
    "7. Jangan membuka berkas kunci, skor algoritma, atau data pengaju selama menilai.",
    "8. Simpan berkas dengan nama penilaian_penilai{1|2}_<inisial>.xlsx lalu serahkan ke peneliti.",
]
thin = Side(style="thin", color="999999")
def tulis_xlsx(path, data, seed, judul):
    r = np.random.default_rng(seed)
    d = data.iloc[r.permutation(len(data))].reset_index(drop=True).copy()
    swap = r.random(len(d)) < 0.5
    j1 = np.where(swap, d.judul_2, d.judul_1); j2 = np.where(swap, d.judul_1, d.judul_2)
    wb = Workbook(); ws0 = wb.active; ws0.title = "Petunjuk"
    for i, t in enumerate(PETUNJUK, 1):
        ws0.cell(i, 1, t).font = Font(bold=(i == 1), size=12 if i == 1 else 11)
    ws0.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Penilaian")
    head = ["no", "judul_1", "judul_2", "label", "yakin", "catatan"]  # label: 2/1/0
    ws.append(head)
    for a, b, c in zip(d.no, j1, j2):
        ws.append([a, b, c, None, None, None])
    fill = PatternFill("solid", fgColor="1F4E78")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = fill; c.alignment = Alignment(horizontal="center", vertical="center")
    for w, col in zip([8, 60, 60, 13, 9, 30], "ABCDEF"):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top"); c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
    n = len(d) + 1
    dv1 = DataValidation(type="list", formula1='"2,1,0"', allow_blank=True, showErrorMessage=True,
                         errorTitle="Label tidak valid", error="Isi 2 (mirip), 1 (mirip sebagian), atau 0 (tidak mirip)")
    dv2 = DataValidation(type="list", formula1='"ya,ragu"', allow_blank=True)
    ws.add_data_validation(dv1); ws.add_data_validation(dv2)
    dv1.add(f"D2:D{n}"); dv2.add(f"E2:E{n}")
    ws.freeze_panes = "B2"
    ws3 = wb.create_sheet("Identitas")
    for i, t in enumerate(["Lembar", "Inisial penilai", "Peran (mis. peneliti / rekan TI / dosen)", "Tanggal mulai", "Tanggal selesai", "Jumlah baris"], 1):
        ws3.cell(i, 1, t).font = Font(bold=True)
    ws3.cell(1, 2, judul); ws3.cell(6, 2, len(d)); ws3.column_dimensions["A"].width = 40; ws3.column_dimensions["B"].width = 40
    wb.save(path)


tulis_xlsx(os.path.join(out, "penilaian_penilai1_artikel.xlsx"), semua, 101, "Penilai 1, set utama")
tulis_xlsx(os.path.join(out, "penilaian_penilai2_artikel.xlsx"), semua, 202, "Penilai 2, set utama")
tulis_xlsx(os.path.join(out, "penilaian_pilot_artikel.xlsx"), pil, 303, "Pilot 50 pasangan (dua penilai memakai berkas yang sama, diisi terpisah)")

ring = semua.groupby(["sumber_pasangan", "set_klasifikasi"]).size().unstack(fill_value=0)
alg_count = {f"{a}@{s}": int(alami.dipilih_oleh.str.contains(f"{a}@{s}").sum()) for a in ALGORITMA for s in "AB"}
per_q = alami.kueri_terkait.str.split("|").explode().value_counts()
with open(os.path.join(out, "ringkasan_penilaian_artikel.txt"), "w", encoding="utf-8") as f:
    f.write(ring.to_string() + "\n")
    f.write(f"Total baris: {len(semua)} (alami {len(alami)}, buatan {len(art)})\n")
    f.write(f"Pasangan di skenario A (Ekapta-Ekapta): {int(alami.di_skenario_A.sum())}; melibatkan repositori: {int((~alami.di_skenario_A).sum())}\n")
    f.write(f"Lintas set (masuk uji): {int(alami.lintas_set.sum())} | kandidat karya sama: {int(alami.kandidat_karya_sama.sum())}\n")
    f.write(f"Pasangan per kueri: rata-rata {per_q.mean():.1f}, min {per_q.min()}, maks {per_q.max()}\n")
    f.write("Pasangan yang diusulkan tiap algoritma per skenario: " + str(alg_count) + "\n")
    f.write("Perkiraan waktu per penilai (15 detik/pasangan): %.1f jam\n" % (len(semua) * 15 / 3600))
print(open(os.path.join(out, "ringkasan_penilaian_artikel.txt"), encoding="utf-8").read())
