"""Pooling lanjutan: lima teratas BM25 untuk setiap kombinasi k1 dan b dalam ruang penyetelan.

siapkan_penilaian.py mengambil lima teratas BM25 dengan k1 = 1,2 dan b = 0,75. Skrip ini melengkapi pool dengan lima
teratas Jaccard, TF-IDF Cosine, dan ke-35 kombinasi k1 dan b BM25 untuk ke-100 kueri pada skenario A dan B, memakai
aturan pengecualian yang sama dengan evaluasi.py (judul dari pengaju yang sama dan salinan karya sendiri di
repositori). Hasilnya 20 pasangan yang belum ada di lembar utama, dinilai dengan lembar terpisah.

Pakai:
  python siapkan_tambahan_top5.py korpus_master_v4.csv kueri_100.csv kunci_penilaian_artikel.csv \
         karya_sama_ekapta_repo.csv keluaran_dir
Keluaran:
  kunci_tambahan_top5.csv          20 pasangan dalam format kunci
  kunci_penilaian_lengkap.csv      lembar utama + tambahan (1.116 pasangan); dipakai evaluasi.py
  penilaian_tambahan_penilai1.xlsx dan penilaian_tambahan_penilai2.xlsx
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kemiripan import top_k
from kemiripan._umum import bersihkan
from kemiripan.bm25 import bm25_simetris
from kemiripan.jaccard import jaccard
from kemiripan.tfidf_cosine import tfidf_cosine

korpus, kueri, kunci, karya_sama, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
K = pd.read_csv(korpus, dtype=str)
Q = pd.read_csv(kueri, dtype=str)
KU = pd.read_csv(kunci, dtype=str)
L = KU[KU.sumber_pasangan == "pool_3algoritma"]
dinilai = {frozenset(x) for x in zip(L.doc_id_1, L.doc_id_2)}
ks = pd.read_csv(karya_sama, dtype=str)
ks = ks[ks.karya_sama_tinjauan_awal_AI.eq("ya")]
ks_kueri = {}
for a, b in zip(ks.doc_id_ekapta, ks.doc_id_repo):
    ks_kueri.setdefault(a, set()).add(b)
judul = dict(zip(K.doc_id, K.judul_bersih))
qset = dict(zip(Q.doc_id, Q.set))
ekp = set(K.doc_id[K.skenario_A == "True"])
uji = Q.doc_id.tolist()  # semua kueri (tuning dan uji)

baru = {}
for sk, df in (("A", K[K.skenario_A == "True"].reset_index(drop=True)), ("B", K.reset_index(drop=True))):
    tok = bersihkan(df.tokens_std.tolist())
    ids = df.doc_id.values
    pos = {d: i for i, d in enumerate(ids)}
    peng = df.pengaju_kode.fillna("").values
    konfig = [("jaccard", jaccard(tok)), ("tfidf", tfidf_cosine(tok))]
    konfig += [(f"bm25 k1={k1} b={b}", None) for k1, b in itertools.product(np.round(np.arange(0.5, 2.001, 0.25), 2),
                                                                           np.round(np.arange(0.0, 1.001, 0.25), 2))]
    for nama, M in konfig:
        if M is None:
            k1, b = [float(x.split("=")[1]) for x in nama.split()[1:]]
            M = bm25_simetris(tok, k1, b)
        for q in uji:
            i = pos[q]
            ex = ((peng == peng[i]) & (peng != "")) | np.isin(ids, list(ks_kueri.get(q, ())))
            for j in top_k(M, i, 5, ex):
                pas = frozenset((q, ids[j]))
                if pas not in dinilai:
                    baru.setdefault((q, ids[j]), set()).add(f"{nama}@{sk}")
        del M

urut = sorted(baru, key=lambda x: (qset[x[0]] != "uji", x[0], x[1]))   # kueri uji dulu, lalu tuning
T = pd.DataFrame([dict(no=f"T{n:03d}", sumber_pasangan="pool_3algoritma", doc_id_1=q, doc_id_2=d, kueri_terkait=q,
                       set_kueri=qset[q], dipilih_oleh="|".join(sorted(baru[(q, d)])), judul_1=judul[q], judul_2=judul[d],
                       set_klasifikasi=qset[q], lintas_set=False, di_skenario_A=(q in ekp and d in ekp),
                       kandidat_karya_sama=False)
                  for n, (q, d) in enumerate(urut, 1)])
T.to_csv(os.path.join(out, "kunci_tambahan_top5.csv"), index=False)
LENGKAP = pd.concat([KU, T], ignore_index=True)
LENGKAP.to_csv(os.path.join(out, "kunci_penilaian_lengkap.csv"), index=False)

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



tulis_xlsx(os.path.join(out, "penilaian_tambahan_penilai1.xlsx"), T, 404, "Penilai 1, pasangan tambahan")
tulis_xlsx(os.path.join(out, "penilaian_tambahan_penilai2.xlsx"), T, 505, "Penilai 2, pasangan tambahan")
print(f"Pasangan tambahan: {len(T)} ({T.set_klasifikasi.value_counts().to_dict()}); kunci lengkap: {len(LENGKAP)} pasangan")
print(LENGKAP.groupby(["sumber_pasangan", "set_klasifikasi"]).size().unstack(fill_value=0).to_string())
print(T.dipilih_oleh.str.split("|").explode().str.split(" ").str[0].str.split("@").str[0].value_counts().to_string())
