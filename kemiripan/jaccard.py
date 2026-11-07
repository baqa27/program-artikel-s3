"""Jaccard Similarity, Persamaan (1) artikel: J(A, B) = |A ∩ B| / |A ∪ B|.

A dan B adalah himpunan kata unik dua judul. Nilai 0 sampai 1.

Catatan penerapan ke Ekapta (PHP):
    Tidak butuh statistik korpus. Cukup pecah dua judul menjadi kata, buang duplikat,
    lalu hitung irisan dan gabungan. Di PHP: array_unique, array_intersect, array_merge.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ._umum import pecah


def jaccard(tokens: list[str]) -> np.ndarray:
    """Matriks Jaccard n x n untuk semua pasangan judul di korpus."""
    cv = CountVectorizer(tokenizer=pecah, lowercase=False, token_pattern=None, binary=True)
    B = cv.fit_transform(tokens).astype(np.float64)
    inter = (B @ B.T).toarray()
    size = np.asarray(B.sum(axis=1)).ravel()
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return J


def jaccard_dua_judul(a: str, b: str) -> float:
    """Jaccard untuk satu pasangan judul (string token)."""
    A, B = set(pecah(a)), set(pecah(b))
    return len(A & B) / len(A | B) if A | B else 0.0


def skor_judul_baru(korpus: list[str], pasangan: list[tuple[str, int]]) -> np.ndarray:
    """Skor judul di luar korpus terhadap judul korpus. pasangan = [(token_judul_baru, indeks_korpus)]."""
    return np.array([jaccard_dua_judul(baru, korpus[j]) for baru, j in pasangan])
