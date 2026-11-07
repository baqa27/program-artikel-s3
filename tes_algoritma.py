"""Tes kecil untuk folder kemiripan/. Jalankan: python tes_algoritma.py"""
import numpy as np
from kemiripan import jaccard, tfidf_cosine, bm25_simetris, bm25_mentah, skor_judul_baru, hitung_semua

DOK = ["sistem pakar diagnosa sakit sapi naive bayes",
       "sistem pakar diagnosa sakit kambing naive bayes",
       "prediksi harga saham lstm",
       "sistem pakar diagnosa sakit sapi naive bayes"]

def test_jaccard_hitung_tangan():
    J = jaccard(DOK)
    assert abs(J[0, 1] - 6 / 8) < 1e-12          # irisan 6 kata, gabungan 8 kata
    assert J[0, 2] == 0 and abs(J[0, 3] - 1) < 1e-12

def test_tfidf_rentang():
    T = tfidf_cosine(DOK)
    assert np.allclose(np.diag(T), 1) and T.min() >= -1e-12 and T.max() <= 1 + 1e-9

def test_bm25_simetris_dan_diri():
    S = bm25_simetris(DOK)
    assert np.allclose(S, S.T) and np.allclose(np.diag(S), 1)
    assert S[0, 3] > S[0, 1] > S[0, 2]

def test_bm25_rumus_satu_sel():
    n, k1, b = 4, 1.2, 0.75
    dl = np.array([7, 7, 4, 7]); avg = dl.mean()
    idf = np.log((n - 1 + 0.5) / (1 + 0.5) + 1)
    tf = 1 * (k1 + 1) / (1 + k1 * (1 - b + b * 4 / avg))
    assert abs(bm25_mentah(DOK)[2, 2] - 4 * idf * tf) < 1e-9

def test_judul_baru_identik_dan_urutan_kata():
    baru = [(DOK[0], 0), ("bayes naive sapi sakit diagnosa pakar sistem", 0)]
    for alg, v in skor_judul_baru(DOK, baru).items():
        assert np.allclose(v, 1, atol=1e-6), alg

def test_judul_baru_sama_dengan_matriks_korpus():
    M = hitung_semua(DOK)
    S = skor_judul_baru(DOK, [(DOK[1], 0)])
    for alg in ["jaccard", "tfidf", "bm25"]:
        assert abs(S[alg][0] - M[alg][1, 0]) < 1e-9, alg

if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f(); print("lulus:", f.__name__)
