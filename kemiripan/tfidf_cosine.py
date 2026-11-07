"""TF-IDF Cosine, Persamaan (2)-(3) artikel.

Bobot kata = tf x idf, dengan idf = ln[(1 + N) / (1 + df)] + 1 (bawaan scikit-learn, smooth_idf=True).
Setiap vektor judul dinormalkan (norma L2), jadi cosine = jumlah perkalian bobot kata yang sama.

Catatan penerapan ke Ekapta (PHP):
    Simpan tabel kosakata berisi kata dan nilai idf-nya (satu baris per kata) di MySQL.
    Untuk judul baru: hitung tf tiap kata, kalikan idf, normalkan vektor, lalu jumlahkan
    perkalian bobot dengan vektor judul lama (bisa disimpan juga). Kata yang tidak ada di
    kosakata diabaikan, sama seperti di scikit-learn. Tabel idf diperbarui bila korpus berubah.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ._umum import pecah


def buat_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=pecah, lowercase=False, token_pattern=None)


def tfidf_matrix(tokens: list[str]):
    """Matriks TF-IDF (judul x kata), sudah dinormalkan L2."""
    return buat_vectorizer().fit_transform(tokens)


def tfidf_cosine(tokens: list[str]) -> np.ndarray:
    """Matriks cosine TF-IDF n x n."""
    X = tfidf_matrix(tokens)
    return (X @ X.T).toarray()


def skor_judul_baru(korpus: list[str], pasangan: list[tuple[str, int]]) -> np.ndarray:
    """Skor judul baru dengan statistik korpus yang dibekukan (kosakata dan idf dari korpus saja)."""
    tv = buat_vectorizer()
    X = tv.fit_transform(korpus)
    out = []
    for baru, j in pasangan:
        xa = tv.transform([baru])
        out.append(float((xa @ X[j].T).toarray()[0, 0]))
    return np.array(out)
