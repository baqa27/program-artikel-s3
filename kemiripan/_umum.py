"""Fungsi bantu yang dipakai bersama oleh ketiga algoritma dan pembanding SQL LIKE.

Masukan semua algoritma adalah daftar string token hasil praproses (kolom tokens_std pada
korpus_master), dengan kata dipisah spasi. Contoh: "sistem pakar diagnosa rusa sepeda motor".
"""
from __future__ import annotations


def pecah(s: str) -> list[str]:
    """Memecah string token menjadi daftar kata. Praproses sudah selesai sebelumnya."""
    return s.split()


def bersihkan(tokens: list) -> list[str]:
    """Mengganti nilai kosong (NaN atau None) dengan string kosong."""
    return [t if isinstance(t, str) else "" for t in tokens]
