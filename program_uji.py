"""Program uji berbasis web: lima judul paling mirip menurut Jaccard, TF-IDF Cosine, dan BM25.

Pakai:
  pip install flask
  python program_uji.py korpus_master_v4.csv [tabel6_parameter.csv]
lalu buka http://127.0.0.1:5000 di peramban.

- Judul masukan melewati praproses yang sama dengan korpus (kemiripan/praproses.py).
- Statistik korpus (IDF, avgdl) dibekukan: judul masukan tidak mengubahnya.
- Parameter k1, b, dan ambang dibaca dari tabel6_parameter.csv hasil evaluasi.py. Tanpa berkas itu,
  program memakai k1 = 1,2, b = 0,75, dan tidak menandai ambang.
- Program hanya untuk tangkapan layar artikel dan demonstrasi; tidak terhubung ke Ekapta.
"""
from __future__ import annotations

import html
import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kemiripan import skor_judul_baru
from kemiripan._umum import bersihkan
from kemiripan.praproses import praproses

NAMA = {"jaccard": "Jaccard", "tfidf": "TF-IDF Cosine", "bm25": "BM25"}
K = pd.read_csv(sys.argv[1], dtype=str)
KORPUS = {"A": K[K.skenario_A == "True"].reset_index(drop=True), "B": K.reset_index(drop=True)}
PARAM = {}
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    for r in pd.read_csv(sys.argv[2]).itertuples():
        PARAM[(r.skenario, r.algoritma)] = dict(k1=r.k1, b=r.b, ambang=r.ambang)

app = Flask(__name__)

HALAMAN = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Program Uji Kemiripan Judul</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1b1b1b;background:#fafafa}}
h1{{font-size:20px;margin:0 0 4px}} p.sub{{margin:0 0 16px;color:#555;font-size:13px}}
form{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}
input[type=text]{{flex:1;min-width:280px;padding:8px;font-size:14px;border:1px solid #bbb;border-radius:4px}}
select,button{{padding:8px 12px;font-size:14px;border-radius:4px;border:1px solid #bbb}}
button{{background:#1f4e78;color:#fff;border-color:#1f4e78;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.kartu{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px}}
.kartu h2{{font-size:15px;margin:0 0 6px}} .param{{font-size:12px;color:#666;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #eee;padding:5px;text-align:left;vertical-align:top}}
td.s{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}} tr.lewat td{{background:#fdecea}}
.token{{font-size:12px;color:#555;margin-bottom:12px}}
</style></head><body>
<h1>Program Uji Kemiripan Judul Tugas Akhir</h1>
<p class="sub">Lima judul paling mirip menurut Jaccard, TF-IDF Cosine, dan BM25. Baris merah: skor di atas ambang hasil penyetelan.</p>
<form method="get">
<input type="text" name="judul" placeholder="Tulis judul tugas akhir" value="{judul}">
<select name="skenario"><option value="A"{selA}>Skenario A: 354 judul Ekapta</option>
<option value="B"{selB}>Skenario B: 5.229 judul Ekapta + repositori</option></select>
<button type="submit">Periksa</button></form>
{isi}</body></html>"""


def kartu(alg, skor, df, sk):
    p = PARAM.get((sk, alg), {})
    amb = p.get("ambang")
    urut = np.argsort(-skor, kind="stable")[:5]
    baris = []
    for j in urut:
        kelas = ' class="lewat"' if amb is not None and skor[j] >= amb else ""
        baris.append(f"<tr{kelas}><td>{html.escape(df.judul_bersih[j])}<br><small>{df.doc_id[j]}</small></td>"
                     f"<td class='s'>{skor[j]:.3f}".replace(".", ",") + "</td></tr>")
    info = []
    if alg == "bm25":
        info.append(f"k1 = {p.get('k1', 1.2)}, b = {p.get('b', 0.75)}")
    info.append(f"ambang = {amb}" if amb is not None else "ambang belum disetel")
    return (f"<div class='kartu'><h2>{NAMA[alg]}</h2><div class='param'>{'; '.join(info)}</div>"
            f"<table><tr><th>Judul</th><th>Skor</th></tr>{''.join(baris)}</table></div>")


@app.route("/")
def beranda():
    judul = request.args.get("judul", "").strip()
    sk = request.args.get("skenario", "A")
    isi = ""
    if judul:
        df = KORPUS[sk]
        tok = bersihkan(df.tokens_std.tolist())
        q = praproses(judul)
        pas = [(q, j) for j in range(len(df))]
        p = PARAM.get((sk, "bm25"), {})
        S = skor_judul_baru(tok, pas, k1=float(p.get("k1", 1.2)), b=float(p.get("b", 0.75)))
        isi = (f"<div class='token'>Token hasil praproses: <b>{html.escape(q) or '(kosong)'}</b></div>"
               f"<div class='grid'>{''.join(kartu(a, np.asarray(S[a]), df, sk) for a in ('jaccard', 'tfidf', 'bm25'))}</div>")
    return HALAMAN.format(judul=html.escape(judul), selA=" selected" if sk == "A" else "",
                          selB=" selected" if sk == "B" else "", isi=isi)


if __name__ == "__main__":
    app.run(debug=False)
