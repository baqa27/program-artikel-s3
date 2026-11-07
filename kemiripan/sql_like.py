"""Pembanding dasar: meniru pemeriksaan Check Plagiarism yang berjalan di Ekapta.

Kueri di Ekapta (Laravel):
    Pengajuan::where('judul', 'LIKE', '%' . $judul . '%')

Artinya, sebuah judul lain dianggap cocok bila memuat SELURUH teks judul baru.
Tanda % adalah wildcard SQL (boleh ada teks apa saja), bukan persentase.
Di dalam pola LIKE, karakter '_' pada judul juga menjadi wildcard satu karakter.

Asumsi collation: utf8mb4_unicode_ci (bawaan Laravel), yaitu tidak membedakan huruf
kapital dan aksen. Cek config/database.php Ekapta; bila collation berbeda, ubah
fungsi _normal().

Fungsi ini memakai judul asli (kolom judul_asli), tanpa praproses, karena Ekapta
juga membandingkan teks judul apa adanya. Hasilnya biner (cocok atau tidak), jadi
hanya precision, recall, dan F1-score yang dihitung, tanpa nDCG@5.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np


def _normal(teks: str) -> str:
    teks = unicodedata.normalize("NFKD", str(teks))
    teks = "".join(c for c in teks if not unicodedata.combining(c))
    return teks.casefold()


def _pola_like(judul: str) -> re.Pattern:
    """Ubah '%judul%' menjadi regex: % -> .*, _ -> ., karakter lain literal."""
    bagian = []
    for c in _normal(judul):
        if c == "%":
            bagian.append(".*")
        elif c == "_":
            bagian.append(".")
        else:
            bagian.append(re.escape(c))
    return re.compile(".*" + "".join(bagian) + ".*", re.DOTALL)


def sql_like_cocok(judul_baru: str, judul_lain: str) -> bool:
    """True bila judul_lain cocok dengan pola '%judul_baru%'."""
    return _pola_like(judul_baru).fullmatch(_normal(judul_lain)) is not None


def sql_like_matrix(judul: list[str]) -> np.ndarray:
    """Matriks n x n berisi 1.0 bila judul kolom cocok dengan pola judul baris.

    Tidak simetris: M[i, j] = 1 artinya judul j memuat seluruh teks judul i.
    Diagonal diisi 0 karena Ekapta membandingkan judul baru dengan judul lain.
    """
    pola = [_pola_like(j) for j in judul]
    teks = [_normal(j) for j in judul]
    n = len(judul)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and pola[i].fullmatch(teks[j]):
                M[i, j] = 1.0
    return M


def warna_ekapta(jumlah_cocok: int) -> str:
    """Peringatan warna di Ekapta: hijau untuk 0-1 judul cocok, kuning untuk 2 atau lebih."""
    return "kuning" if jumlah_cocok >= 2 else "hijau"
