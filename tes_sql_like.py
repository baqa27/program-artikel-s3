"""Tes pembanding SQL LIKE. Jalankan: python tes_sql_like.py"""
from kemiripan import sql_like_cocok, sql_like_matrix, warna_ekapta

A = "Sistem Pakar Diagnosa Penyakit Sapi Metode Naive Bayes"

def test_sama_persis_dan_kapital():
    assert sql_like_cocok(A, A.upper())

def test_judul_memuat_judul_baru():
    assert sql_like_cocok(A, A + " Berbasis Web")

def test_beda_satu_kata_lolos():
    assert not sql_like_cocok(A, A.replace("Sapi", "Kambing"))

def test_aksen():
    assert sql_like_cocok("Metode Naive Bayes", "Metode Naïve Bayes")

def test_matriks_tidak_simetris():
    M = sql_like_matrix([A, A + " Berbasis Web"])
    assert M[0, 1] == 1 and M[1, 0] == 0 and M[0, 0] == 0

def test_warna():
    assert warna_ekapta(1) == "hijau" and warna_ekapta(2) == "kuning"

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f()
    print("semua tes SQL LIKE lulus")
