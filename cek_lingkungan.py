"""Jalankan DI LAPTOP PENELITI (Lenovo LOQ), di lingkungan Python yang dipakai eksperimen.
Mengumpulkan bukti untuk penanda M17-M26 di Tabel 8.4 dan 8.5 proposal.

    python cek_lingkungan.py  [folder_kode_proyek]

Keluaran: folder bukti_laptop/ berisi lingkungan.txt dan requirements.txt (pip freeze).
Kirim isi lingkungan.txt ke Claude atau salin manual ke tabel.
"""
import datetime, importlib.metadata as md, os, platform, re, shutil, subprocess, sys

proj = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs("bukti_laptop", exist_ok=True)
L = []
def tulis(k, v): L.append(f"{k}: {v}")

tulis("Waktu", datetime.datetime.now().isoformat(timespec="seconds"))
tulis("M17 Sistem operasi", f"{platform.system()} {platform.release()} (versi {platform.version()})")
if platform.system() == "Windows":
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        tulis("M17 Windows (Caption)", out)
    except Exception as e:
        tulis("M17 Windows (Caption)", f"gagal: {e}")
if "microsoft" in platform.release().lower():
    tulis("Catatan", "Kode berjalan di WSL. Catat juga versi Windows induknya.")
tulis("M18 Python", sys.version.split()[0])
tulis("Lingkungan Python (sys.prefix)", sys.prefix)
tulis("Nama venv/conda", os.environ.get("CONDA_DEFAULT_ENV") or os.path.basename(os.environ.get("VIRTUAL_ENV", "")) or "(tidak terdeteksi)")

def cpu():
    s = platform.system()
    try:
        if s == "Windows":
            return subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=60).stdout.strip()
        if s == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if s == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return f"gagal: {e}"
tulis("M26 Prosesor", cpu())
try:
    if platform.system() == "Windows":
        ram = subprocess.run(["powershell", "-NoProfile", "-Command",
              "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
              capture_output=True, text=True, timeout=60).stdout.strip() + " GB"
    else:
        ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3:.1f} GB"
except Exception as e:
    ram = f"gagal: {e}"
tulis("RAM terbaca sistem (Tabel 8.5 menulis 20 GB)", ram)
du = shutil.disk_usage(os.path.abspath(os.sep))
tulis("Kapasitas disk sistem (Tabel 8.5 menulis 1 TB)", f"{du.total / 1000**3:.0f} GB (desimal)")

PUSTAKA = {"M20": ["PySastrawi"], "M21": ["nltk", "snowballstemmer", "PyStemmer"],
           "M22": ["numpy", "scipy", "scikit-learn"], "M23": ["pandas", "matplotlib"],
           "M24": ["rank-bm25", "rank_bm25", "bm25s", "gensim"], "lain": ["openpyxl", "python-docx"]}
for m, pk in PUSTAKA.items():
    for p in pk:
        try:
            tulis(f"{m} {p}", md.version(p))
        except md.PackageNotFoundError:
            tulis(f"{m} {p}", "tidak terpasang")

# pustaka yang benar-benar diimpor kode proyek
impor = {}
for root, _, files in os.walk(proj):
    if any(x in root for x in ("venv", ".venv", "site-packages", ".git")):
        continue
    for f in files:
        if f.endswith((".py", ".ipynb")):
            try:
                teks = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            mods = set()
            if f.endswith(".py"):
                import ast
                try:
                    for node in ast.walk(ast.parse(teks)):
                        if isinstance(node, ast.Import):
                            mods |= {a.name.split(".")[0] for a in node.names}
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            mods.add(node.module.split(".")[0])
                except SyntaxError:
                    pass
            else:
                mods |= set(re.findall(r"(?:from|import)\s+([A-Za-z_]\w*)", teks))
            for mod in mods:
                impor.setdefault(mod, set()).add(f)
tulis("Modul yang diimpor kode proyek", "; ".join(f"{k} ({', '.join(sorted(v))})" for k, v in sorted(impor.items())))
tulis("M19 usulan lingkungan kerja", "Jupyter Notebook (ada .ipynb)" if any(f.endswith(".ipynb") for _, _, fs in os.walk(proj) for f in fs) else "skrip .py (VS Code atau editor lain, tegaskan sendiri)")

with open("bukti_laptop/lingkungan.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
open("bukti_laptop/requirements.txt", "w", encoding="utf-8").write(freeze)
print("\n".join(L)); print("\nTersimpan: bukti_laptop/lingkungan.txt dan bukti_laptop/requirements.txt")
