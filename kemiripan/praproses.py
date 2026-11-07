"""Praproses judul: rekonstruksi langkah yang menghasilkan kolom tokens_std pada korpus.

Dipakai untuk judul di luar korpus (pasangan buatan, judul masukan program uji, dan penerapan di Ekapta).
Urutan langkah:
    1. normalisasi Unicode (NFKD, aksen dibuang) dan case folding
    2. apostrof dibuang (qur'an -> quran), titik di dalam kata/angka dibuang (c4.5 -> c45),
       e-commerce -> ecommerce, tanda baca lain menjadi spasi
    3. penyeragaman ejaan: website -> web
    4. tokenisasi dengan spasi
    5. buang angka tahun (1900-2099) dan stopword pada kata asli (sebelum stemming)
    6. stemming: Sastrawi untuk judul Indonesia/campuran, Snowball "english" untuk judul Inggris;
       token beraksara Arab dibiarkan apa adanya

Hasil fungsi ini dicocokkan dengan tokens_std korpus lewat cek_praproses.py; persentase kecocokan
dicatat di bukti/cek_praproses.txt. Token yang berbeda harus dilaporkan, bukan disembunyikan.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Stopword bahasa Inggris yang terbukti dibuang pada korpus (lihat cek_praproses.py)
STOP_EN_DI_JUDUL_ID = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "the", "to", "with", "using"}
STOP_JUDUL_EN = STOP_EN_DI_JUDUL_ID | {"through", "as", "based", "toward", "towards", "will", "we", "are",
                                      "not", "between", "under", "when", "where"}
STOP_DOMAIN_ID = {"menggunakan"}
TAHUN = re.compile(r"(19|20)\d\d")
EJAAN = {"website": "web"}


@lru_cache(maxsize=1)
def _sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    return StemmerFactory().create_stemmer(), set(StopWordRemoverFactory().get_stop_words())


@lru_cache(maxsize=1)
def _snowball():
    import snowballstemmer
    return snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def _stem_id(w: str) -> str:
    return _sastrawi()[0].stem(w)


def normalisasi(judul: str) -> str:
    s = unicodedata.normalize("NFKD", judul)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"['’‘`´]", "", s)
    s = re.sub(r"(?<=\w)\.(?=\w)", "", s)
    s = re.sub(r"\be-commerce\b", "ecommerce", s)
    s = re.sub(r"[^\w]+|_", " ", s)
    kata = [EJAAN.get(w, w) for w in s.split()]
    return " ".join(kata)


def token_dari_norm(norm: str, bahasa: str = "Indonesia") -> list[str]:
    out = []
    if bahasa == "English":
        st = _snowball()
        for w in norm.split():
            if TAHUN.fullmatch(w) or not w.isascii() or w in STOP_JUDUL_EN:
                continue
            out.append(st.stemWord(w))
        return out
    sw = _sastrawi()[1]
    for w in norm.split():
        if TAHUN.fullmatch(w) or not w.isascii():
            continue
        if w in sw or w in STOP_EN_DI_JUDUL_ID or w in STOP_DOMAIN_ID:
            continue
        s = _stem_id(w)
        if s:
            out.append(s)
    return out


def praproses(judul: str, bahasa: str = "Indonesia") -> str:
    """Judul mentah -> string token (format sama dengan kolom tokens_std)."""
    return " ".join(token_dari_norm(normalisasi(judul), bahasa))
