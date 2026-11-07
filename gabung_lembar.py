"""Menggabungkan lembar utama dan lembar tambahan milik satu penilai menjadi satu berkas (1.116 baris).

Pakai (setelah kedua lembar terisi):
  python gabung_lembar.py penilai1_utama.xlsx penilai1_tambahan.xlsx penilai1_lengkap.xlsx

Isi baris tidak diubah: baris lembar tambahan (nomor T001-T020) ditambahkan di bawah baris lembar utama.
Lembar Petunjuk dan Identitas diambil dari lembar utama; jumlah baris di Identitas diperbarui.
Berkas hasil bisa langsung dipakai hitung_kappa.py dan gabung_label.py tanpa tanda +.
"""
import sys
from copy import copy

from openpyxl import load_workbook

utama, tambahan, keluar = sys.argv[1:4]
wb = load_workbook(utama)
ws = wb["Penilaian"]
wt = load_workbook(tambahan)["Penilaian"]
nomor = {ws.cell(r, 1).value for r in range(2, ws.max_row + 1)}
contoh = [ws.cell(2, c) for c in range(1, 7)]
n_awal = ws.max_row - 1
for row in wt.iter_rows(min_row=2, values_only=True):
    if row[0] in nomor:
        sys.exit(f"Nomor {row[0]} sudah ada di lembar utama.")
    ws.append(list(row[:6]))
    for c, ref in zip(ws[ws.max_row], contoh):
        c.alignment = copy(ref.alignment); c.border = copy(ref.border)
kosong = sum(1 for r in range(2, ws.max_row + 1) if ws.cell(r, 4).value in (None, ""))
wb["Identitas"].cell(6, 2, ws.max_row - 1)
wb.save(keluar)
print(f"{n_awal} baris utama + {ws.max_row - 1 - n_awal} baris tambahan = {ws.max_row - 1} baris -> {keluar}")
if kosong:
    print(f"PERINGATAN: {kosong} baris belum berlabel.")
