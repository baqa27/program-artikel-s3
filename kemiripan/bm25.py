"""BM25 simetris, Persamaan (4)-(6) artikel. Ditulis sendiri tanpa pustaka BM25.

Sumbangan kata q dari kueri ke judul d:
    idf(q) x f(q, d) x (k1 + 1) / [f(q, d) + k1 x (1 - b + b x |d| / avgdl)]
    idf(q) = ln[(N - df + 0.5) / (df + 0.5) + 1]
Kata kueri dihitung sekali (himpunan). Skor simetris:
    s(a, b) = 1/2 x [BM25(a, b) / BM25(a, a) + BM25(b, a) / BM25(b, b)]

Catatan penerapan ke Ekapta (PHP):
    Simpan df tiap kata, N (jumlah judul), dan avgdl (rata-rata panjang judul) di MySQL.
    Rumus di atas hanya memakai penjumlahan, perkalian, pembagian, dan log, jadi bisa
    ditulis langsung di PHP. Nilai k1 dan b memakai hasil penyetelan pada set tuning.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from ._umum import pecah


def bm25_mentah(tokens: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Matriks BM25 asimetris. Baris = kueri, kolom = dokumen."""
    cv = CountVectorizer(tokenizer=pecah, lowercase=False, token_pattern=None)
    C = cv.fit_transform(tokens).tocsr().astype(np.float64)
    n = C.shape[0]
    dl = np.asarray(C.sum(axis=1)).ravel()
    avgdl = dl.mean() if n else 0.0
    df = np.asarray((C > 0).sum(axis=0)).ravel()
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
    Cc = C.tocoo()
    denom = Cc.data + k1 * (1 - b + b * dl[Cc.row] / avgdl)
    W = sparse.csr_matrix((Cc.data * (k1 + 1) / denom, (Cc.row, Cc.col)), shape=C.shape)
    Q = (C > 0).astype(np.float64).multiply(idf).tocsr()
    return (Q @ W.T).toarray()


def bm25_simetris(tokens: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Matriks BM25 simetris n x n (Persamaan 6 artikel)."""
    R = bm25_mentah(tokens, k1, b)
    d = np.diag(R).copy()
    d[d == 0] = 1.0
    return 0.5 * (R / d[:, None] + R.T / d[None, :])


class StatistikKorpus:
    """Statistik korpus yang dibekukan untuk menilai judul baru: N, avgdl, dan df tiap kata."""

    def __init__(self, korpus: list[str]):
        cv = CountVectorizer(tokenizer=pecah, lowercase=False, token_pattern=None)
        C = cv.fit_transform(korpus)
        self.n = C.shape[0]
        self.avgdl = float(np.asarray(C.sum(axis=1)).ravel().mean())
        self.df = dict(zip(cv.get_feature_names_out(), np.asarray((C > 0).sum(axis=0)).ravel()))

    def idf(self, w: str) -> float:
        d = self.df.get(w, 0)
        return np.log((self.n - d + 0.5) / (d + 0.5) + 1.0)

    def skor(self, q_tok: list[str], d_tok: list[str], k1: float, b: float) -> float:
        tf: dict[str, int] = {}
        for w in d_tok:
            tf[w] = tf.get(w, 0) + 1
        dl = len(d_tok)
        s = 0.0
        for w in set(q_tok):
            f = tf.get(w, 0)
            if f:
                s += self.idf(w) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
        return s


def skor_judul_baru(korpus: list[str], pasangan: list[tuple[str, int]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Skor BM25 simetris judul baru terhadap judul korpus. Kata baru diberi df = 0."""
    st = StatistikKorpus(korpus)
    out = []
    for baru, j in pasangan:
        a, d = pecah(baru), pecah(korpus[j])
        saa = st.skor(a, a, k1, b) or 1.0
        sdd = st.skor(d, d, k1, b) or 1.0
        out.append(0.5 * (st.skor(a, d, k1, b) / saa + st.skor(d, a, k1, b) / sdd))
    return np.array(out)
