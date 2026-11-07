"""Menjalankan ketiga algoritma sekaligus dan mengambil judul teratas.

Contoh:
    from kemiripan import hitung_semua, top_k
    S = hitung_semua(df.tokens_std.tolist(), k1=1.2, b=0.75)
    top_k(S["bm25"], i=0, k=5)
"""
from __future__ import annotations

import numpy as np

from ._umum import bersihkan
from .bm25 import bm25_simetris
from .bm25 import skor_judul_baru as bm25_baru
from .jaccard import jaccard
from .jaccard import skor_judul_baru as jaccard_baru
from .tfidf_cosine import skor_judul_baru as tfidf_baru
from .tfidf_cosine import tfidf_cosine

ALGORITMA = ("jaccard", "tfidf", "bm25")


def hitung_semua(tokens: list[str], k1: float = 1.2, b: float = 0.75) -> dict:
    """Matriks kemiripan n x n untuk ketiga algoritma."""
    tokens = bersihkan(tokens)
    return {
        "jaccard": jaccard(tokens),
        "tfidf": tfidf_cosine(tokens),
        "bm25": bm25_simetris(tokens, k1, b),
    }


def top_k(S: np.ndarray, i: int, k: int = 5, kecualikan: np.ndarray | None = None) -> np.ndarray:
    """Indeks k judul teratas untuk kueri i. Judul itu sendiri selalu dilewati;
    `kecualikan` = mask boolean judul lain yang dilewati (misalnya pengaju yang sama)."""
    s = S[i].astype(np.float64).copy()
    s[i] = -np.inf
    if kecualikan is not None:
        s[kecualikan] = -np.inf
    return np.argsort(-s, kind="stable")[:k]


def skor_judul_baru(korpus: list[str], pasangan: list[tuple[str, int]], k1: float = 1.2, b: float = 0.75) -> dict:
    """Skor judul di luar korpus (pasangan buatan, program uji) untuk ketiga algoritma.
    Statistik korpus dibekukan; judul baru tidak mengubahnya."""
    korpus = bersihkan(korpus)
    return {
        "jaccard": jaccard_baru(korpus, pasangan),
        "tfidf": tfidf_baru(korpus, pasangan),
        "bm25": bm25_baru(korpus, pasangan, k1, b),
    }
