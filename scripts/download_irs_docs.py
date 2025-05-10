"""
Download *all* IRS PDFs for a single year.

• 2024 ⇒ grabs everything in /pub/irs‑pdf/      (≈17 k files, ~4 GB)
• 2023‑1900 ⇒ grabs every file that ends "--YYYY.pdf" in /pub/irs‑prior/
"""
import os, re, time, requests, pathlib, sys, json

# ---------- user‑adjustable ----------
YEAR       = "2024"                 # "2024", "2023", "2018", ...

# Determine SAVE_ROOT relative to the script's location
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SAVE_ROOT  = SCRIPT_DIR.parent / "data" / "raw_pdfs"

HEADERS = {                         # avoid 403/slow‑downs
    "User-Agent": "Mozilla/5.0 (compatible; IRS-scraper/1.0; +mailto:you@example.com)",
    "Accept-Encoding": "gzip, deflate",
}
# ------------------------------------

if YEAR == "2024":                             # current season
    BASE_URL   = "https://www.irs.gov/pub/irs-pdf/"
    YEAR_REGEX = re.compile(r'^.+\.pdf$', re.I) # everything
else:                                           # prior seasons
    BASE_URL   = "https://www.irs.gov/pub/irs-prior/"
    YEAR_REGEX = re.compile(fr'^.+--{YEAR}\.pdf$', re.I)  # only that year

dest_dir = pathlib.Path(SAVE_ROOT) / YEAR
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Fetching directory listing for {YEAR} …")
try:
    html = requests.get(BASE_URL, headers=HEADERS, timeout=30).text
except requests.exceptions.RequestException as e:
    sys.exit(f"Could not reach {BASE_URL} → {e}")

# Harvest every href that ends .pdf and passes the year filter
candidates = re.findall(r'href="([^"+]+\.pdf)"', html, flags=re.I)
pdfs       = [p.split('/')[-1] for p in candidates if YEAR_REGEX.search(p)]

# For testing, limit to first 5 PDFs
#pdfs = pdfs[:5] # For testing, limit to first 5 PDFs

if not pdfs:
    sys.exit(f"No PDFs matched year {YEAR}.  The naming convention may have changed.")

print(f"→ {len(pdfs)} PDFs match year {YEAR}.  Starting download …")
manifest = []
for fname in pdfs:
    url  = BASE_URL + fname
    path = dest_dir / fname
    if path.exists():
        continue
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        manifest.append({"year": YEAR, "url": url, "local": str(path)})
        print(f"  ✓ {fname}")
        time.sleep(0.3)             # be gentle – IRS asks bots to throttle
    except Exception as e:
        print(f"  ✗ {fname}: {e}")

# Save a companion manifest you can feed straight into a RAG indexer later
if manifest:
    with open(dest_dir / "manifest.json", "w") as jf:
        json.dump(manifest, jf, indent=2)

print(f"Finished.  Downloaded {len(manifest)} files to {dest_dir}")
