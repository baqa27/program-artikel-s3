"""Simulasi beban penilaian bila jumlah kueri ditambah (dasar keputusan 100 kueri).

Pakai: python simulasi_jumlah_kueri.py <folder 05_kode> korpus_master_v4.csv kueri.csv (60 kueri versi proposal, ada di paket sempro)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from kemiripan.jaccard import jaccard
from kemiripan.tfidf_cosine import tfidf_cosine
from kemiripan.bm25 import bm25_simetris
from kemiripan.pencarian import top_k
from kemiripan._umum import bersihkan
K = pd.read_csv(sys.argv[2])
kq = pd.read_csv(sys.argv[3])
K = K.reset_index(drop=True)
A = K[K.skenario_A == True].reset_index(drop=True)
def pool(df):
    tok = bersihkan(df.tokens_std.tolist())
    mats = {'jaccard': jaccard(tok), 'tfidf': tfidf_cosine(tok), 'bm25': bm25_simetris(tok)}
    ids = df.doc_id.values; peng = df.pengaju_kode.astype(str).values
    out = {}
    for i, did in enumerate(ids):
        if not did.startswith('EKP'): continue
        ex = (peng == peng[i]) & (peng != 'nan')
        s = set()
        for M in mats.values():
            for j in top_k(M, i, 5, ex): s.add(frozenset((did, ids[j])))
        out[did] = s
    return out
pa = pool(A); pb = pool(K)
allq = {q: pa[q] | pb[q] for q in pa}
sizes = pd.Series({q: len(v) for q, v in allq.items()})
print('pasangan unik per kueri (A∪B): mean %.1f median %d' % (sizes.mean(), sizes.median()))
cur = list(kq.doc_id)
base = set().union(*[allq[q] for q in cur])
print('60 kueri sekarang -> pasangan alami (simulasi k1=1.2,b=0.75):', len(base))
rng = np.random.default_rng(42)
pengaju = A.set_index('doc_id').pengaju_kode.astype(str)
used_p = set(pengaju[cur])
rest = [q for q in A.doc_id if q not in cur and pengaju[q] not in used_p]
print('kandidat kueri tersisa (pengaju belum dipakai):', len(rest))
for extra in (20, 40, 60, 90, 140):
    res = []
    for rep in range(20):
        pick = rng.choice(rest, size=min(extra, len(rest)), replace=False)
        s = base.union(*[allq[q] for q in pick])
        res.append(len(s))
    print('total kueri %d: pasangan alami ~%d (+%d), +80 buatan = %d, jam/penilai %.1f' % (60+extra, np.mean(res), np.mean(res)-len(base), np.mean(res)+80, (np.mean(res)+80)*15/3600))
