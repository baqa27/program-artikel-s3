# Kode Eksperimen Artikel: Jaccard, TF-IDF Cosine, dan BM25

Python 3.10 ke atas. Pasang pustaka di satu venv/conda:

```bash
pip install -r requirements.txt
```

Versi pustaka laptop saat replikasi (6 November 2026):
```bash
pip install -r requirements_laptop.txt
```

Jalankan semua perintah dari folder `05_kode`.

## Folder `kemiripan/`

| Berkas | Isi |
|---|---|
| `jaccard.py` | Jaccard Similarity (Persamaan 1 artikel) |
| `tfidf_cosine.py` | TF-IDF Cosine, IDF bawaan scikit-learn (Persamaan 2-3) |
| `bm25.py` | BM25 simetris, kode sendiri tanpa pustaka BM25 (Persamaan 4-6) |
| `praproses.py` | Praproses judul baru (normalisasi, stopword, Sastrawi/Snowball). Cocok 98,9% dengan `tokens_std` judul Indonesia; selisihnya berasal dari koreksi salah ketik manual (`06_bukti/cek_praproses.txt`) |
| `pencarian.py` | `hitung_semua`, `top_k`, `skor_judul_baru` untuk ketiga algoritma |
| `sql_like.py` | Pembanding dasar opsional: meniru Check Plagiarism Ekapta (LIKE) |

LSA sudah dikeluarkan dari paket ini. Kodenya tetap ada di paket sempro bila pembimbing memintanya.

## Urutan kerja

| No | Perintah | Hasil |
|---|---|---|
| 0 | `python tes_algoritma.py` dan `python tes_sql_like.py` | Semua tes lulus |
| 1 | (sudah dijalankan) `python pilih_kueri.py ../04_data/korpus/dataset_ekapta_354.csv ../04_data/ground_truth/buatan_v2_kunci.csv ../04_data/ground_truth/kueri_100.csv` | 100 kueri dipilih dari nol (30 tuning, 70 uji), proporsional per tahun dan prodi |
| 2 | (sudah dijalankan) `python siapkan_penilaian.py ../04_data/korpus/korpus_master_v4.csv ../04_data/ground_truth/kueri_100.csv ../04_data/ground_truth/buatan_v2_kunci.csv ../04_data/ground_truth/near_duplicate.csv ../03_penilaian` | Kunci dan lembar penilaian 1.096 pasangan |
| 2a | (sudah dijalankan) `python siapkan_tambahan_top5.py ../04_data/korpus/korpus_master_v4.csv ../04_data/ground_truth/kueri_100.csv ../03_penilaian/kunci/kunci_penilaian_artikel.csv ../04_data/ground_truth/karya_sama_ekapta_repo.csv ../03_penilaian/kunci` | 20 pasangan dari lima teratas BM25 untuk setiap kombinasi k1 dan b, lembar tambahan kedua penilai, dan `kunci_penilaian_lengkap.csv` (1.116 pasangan; semua kunci ada di `03_penilaian/kunci/`) |
| 3 | Pilot dulu: dua penilai mengisi salinan `03_penilaian/lembar_kosong/pilot_kosong.xlsx`, hitung Kappa pilot dengan `python hitung_kappa.py pilot1.xlsx pilot2.xlsx --keluaran diskusi_pilot.csv`, samakan pedoman. Lalu masing-masing menyalin lembar kosongnya ke `03_penilaian/terisi_penilai1/` atau `terisi_penilai2/` dengan nama `penilaiN_utama.xlsx` dan `penilaiN_tambahan.xlsx`, lalu mengisinya | Pedoman final dan dua lembar terisi |
| 3a | Setelah lembar utama dan tambahan terisi, gabungkan per penilai: `python gabung_lembar.py ../03_penilaian/terisi_penilai1/penilai1_utama.xlsx ../03_penilaian/terisi_penilai1/penilai1_tambahan.xlsx ../03_penilaian/terisi_penilai1/penilai1_lengkap.xlsx` (lakukan juga untuk penilai 2). Berkas `_lengkap` bisa dipakai di langkah 4 dan 5 tanpa tanda `+` | Satu berkas 1.116 baris per penilai |
| 4 | `python hitung_kappa.py ../03_penilaian/terisi_penilai1/penilai1_lengkap.xlsx ../03_penilaian/terisi_penilai2/penilai2_lengkap.xlsx ../03_penilaian/kunci/kunci_penilaian_lengkap.csv` | Kappa 1.116 pasangan (dari label sebelum diskusi) dan `lembar_diskusi.csv` |
| 5 | Kedua penilai membahas pasangan di `03_penilaian/kappa/lembar_diskusi.xlsx` (versi terisi: `lembar_diskusi_terisi.xlsx`) dan mengisi kolom Label Sepakat. Pasangan yang tetap buntu diserahkan ke pembimbing (kolom `label_pembimbing`). Lalu `python gabung_label.py ../03_penilaian/terisi_penilai1/penilai1_lengkap.xlsx ../03_penilaian/terisi_penilai2/penilai2_lengkap.xlsx ../03_penilaian/kappa/lembar_diskusi_terisi.xlsx ../03_penilaian/label_final.csv` | `label_final.csv` 1.116 pasangan (kolom `sumber_label`: sepakat, diskusi, pembimbing) |
| 6 | `python evaluasi.py ../04_data/korpus/korpus_master_v4.csv ../04_data/ground_truth/kueri_100.csv ../03_penilaian/kunci/kunci_penilaian_lengkap.csv ../03_penilaian/label_final.csv ../09_hasil --karya-sama ../04_data/ground_truth/karya_sama_ekapta_repo.csv` (tambah `--sql-like` bila memakai pembanding Ekapta) | Tabel 5-8, sensitivitas, bahan analisis kesalahan, `ringkasan.txt` berdesimal koma. Enam pasangan karya sama (judul Ekapta dan salinannya sendiri di repositori) dikeluarkan, jadi skenario B memakai 342 pasangan tuning dan 768 pasangan uji; `ringkasan.txt` harus menulis cakupan lima teratas 100% |
| 7 | `python program_uji.py ../04_data/korpus/korpus_master_v4.csv ../09_hasil/tabel6_parameter.csv`, buka http://127.0.0.1:5000 | Tangkapan layar Gambar 2 |
| 8 | `python cek_lingkungan.py .` | Bukti versi pustaka untuk Tabel 4 |
| 9 | (sudah dijalankan) `python cek_like_ekapta.py ../04_data/korpus/dataset_ekapta_354.csv` | `06_bukti/cek_like_ekapta.txt`: dasar angka 13 dan 183 pasangan di Pendahuluan |

`evaluasi.py` berjalan sekitar 3 sampai 5 menit dengan 10.000 iterasi. Matriks BM25 skenario B (5.229 x 5.229) dihitung untuk 35 kombinasi k1 dan b satu per satu, jadi memori puncak sekitar 1 GB.

## Catatan

- `simulasi_jumlah_kueri.py` menghasilkan `06_bukti/simulasi_jumlah_kueri.txt`, dasar keputusan 100 kueri. Simulasi itu memakai 60 kueri versi proposal (`kueri.csv` di paket sempro) sebagai titik awal.
- Jalankan ulang langkah 1 dan 2 menghasilkan berkas yang identik (seed 42), sudah dicek.
- Penerapan ke Ekapta (PHP): Jaccard tidak butuh data tambahan; TF-IDF Cosine butuh tabel kosakata dan IDF; BM25 butuh df tiap kata, N, avgdl, k1, dan b. Cocokkan hasil versi PHP dengan versi Python pada judul yang sama.
