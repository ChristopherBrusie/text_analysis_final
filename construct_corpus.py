"""
download_gutenberg_corpus.py

Automatically assembles a 100-text corpus from Project Gutenberg:
  - 50 popular fiction titles
  - 50 popular nonfiction titles

Strategy:
  1. Download the Gutenberg catalog CSV (~70k books, ~10 MB) for metadata.
  2. Scrape gutenberg.org/browse/scores/top for the popularity-ranked ID list.
  3. Walk the ranked list, classify each book as fiction/nonfiction using
     Library of Congress Classification (LoCC) codes + subject keywords.
  4. Download plain-text files for the first 50 of each genre.
  5. Write metadata.csv summarising the corpus.

Usage:
    pip install requests beautifulsoup4
    python download_gutenberg_corpus.py

Output:
    corpus/fiction/        ← 50 .txt files
    corpus/nonfiction/     ← 50 .txt files
    corpus/metadata.csv    ← one row per downloaded book

Gutenberg asks that automated downloads use their mirrors and include
a delay between requests.  This script sleeps 2 s between downloads
and targets the preferred mirror defined in MIRROR below.
"""

import csv
import io
import os
import re
import time
import zipfile

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_PER_GENRE = 50
SLEEP_BETWEEN_DOWNLOADS = 2          # seconds — be polite to Gutenberg
REQUEST_TIMEOUT = 30                 # seconds

# Gutenberg's preferred robot-friendly mirror for bulk downloads
MIRROR = "https://www.gutenberg.org"

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TOP_BOOKS_URL = "https://www.gutenberg.org/browse/scores/top"

OUTPUT_DIR = "corpus"
FICTION_DIR = os.path.join(OUTPUT_DIR, "fiction")
NONFICTION_DIR = os.path.join(OUTPUT_DIR, "nonfiction")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

HEADERS = {"User-Agent": "GutenbergCorpusBuilder/1.0 (text-analysis research)"}

# ---------------------------------------------------------------------------
# Genre classification
# ---------------------------------------------------------------------------

# Library of Congress Classification prefixes that indicate prose fiction
# (excludes poetry P, drama, literary criticism — those land in AMBIGUOUS)
FICTION_LOCC = {
    "PR",   # English literature (novels, short stories)
    "PS",   # American literature
    "PQ",   # Romance literatures (French, Spanish, Italian…)
    "PT",   # Germanic & Scandinavian literature
    "PG",   # Slavic / Baltic / Albanian literature
    "PL",   # East Asian / African / Oceanic literatures
}

# LoCC prefixes that are reliably nonfiction
NONFICTION_LOCC = {
    "A",    # General works / encyclopaedias
    "B",    # Philosophy, psychology, religion
    "C",    # Auxiliary sciences of history
    "D",    # World history
    "E", "F",  # American history
    "G",    # Geography, anthropology, recreation
    "H",    # Social sciences, economics, sociology
    "J",    # Political science
    "K",    # Law
    "L",    # Education
    "M",    # Music (scores — skip in practice)
    "N",    # Fine arts
    "Q",    # Science
    "R",    # Medicine
    "S",    # Agriculture
    "T",    # Technology & engineering
    "U", "V",  # Military / naval science
    "Z",    # Bibliography, library science
}

# Subject-keyword fallback when LoCC is absent or ambiguous
FICTION_SUBJECT_KEYWORDS = [
    "fiction", "novel", "romance", "adventure stories",
    "detective", "mystery", "science fiction", "fantasy",
    "short stories", "tales", "historical fiction",
]

NONFICTION_SUBJECT_KEYWORDS = [
    "history", "biography", "autobiography", "memoir",
    "essay", "essays", "philosophy", "science", "economics",
    "travel", "letters", "correspondence", "political",
    "religion", "self-help", "nature", "medicine",
]

# Subjects that mark poetry / drama — we skip these even inside fiction LoCC
SKIP_SUBJECT_KEYWORDS = [
    "poetry", "poems", "poetic", "drama", "plays", "verse",
    "epic", "sonnets", "ballads",
]


def classify(locc: str, subjects: str) -> str:
    """Return 'fiction', 'nonfiction', or 'skip'."""
    subjects_lower = subjects.lower()
    locc_codes = [c.strip() for c in locc.split(";")] if locc else []

    # Always skip poetry / drama regardless of LoCC
    if any(k in subjects_lower for k in SKIP_SUBJECT_KEYWORDS):
        return "skip"

    # Check LoCC first — most reliable
    for code in locc_codes:
        prefix2 = code[:2]
        prefix1 = code[:1]
        if prefix2 in FICTION_LOCC:
            return "fiction"
        if prefix1 in NONFICTION_LOCC:
            return "nonfiction"

    # Fall back to subject keywords
    if any(k in subjects_lower for k in FICTION_SUBJECT_KEYWORDS):
        return "fiction"
    if any(k in subjects_lower for k in NONFICTION_SUBJECT_KEYWORDS):
        return "nonfiction"

    return "skip"   # can't determine — skip rather than mislabel


# ---------------------------------------------------------------------------
# Step 1: Download and parse the Gutenberg catalog
# ---------------------------------------------------------------------------

def fetch_catalog() -> dict[int, dict]:
    """
    Returns a dict of {gutenberg_id: metadata_dict} for all English texts.
    Catalog columns: Text#, Issued, Title, Language, Authors, Subjects, LoCC, Bookshelves
    """
    print("Downloading Gutenberg catalog CSV (~10 MB)…")
    r = requests.get(CATALOG_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()

    catalog = {}
    reader = csv.DictReader(io.StringIO(r.text))
    for row in reader:
        if row.get("Language", "").strip().lower() != "en":
            continue
        if row.get("Type", "").strip().lower() != "text":
            continue
        try:
            gid = int(row["Text#"])
        except (KeyError, ValueError):
            continue

        catalog[gid] = {
            "id":       gid,
            "title":    row.get("Title", "").strip(),
            "author":   row.get("Authors", "").strip(),
            "issued":   row.get("Issued", "").strip(),
            "subjects": row.get("Subjects", "").strip(),
            "locc":     row.get("LoCC", "").strip(),
            "shelves":  row.get("Bookshelves", "").strip(),
        }

    print(f"  Catalog loaded: {len(catalog):,} English texts")
    return catalog


# ---------------------------------------------------------------------------
# Step 2: Scrape the top-books popularity ranking
# ---------------------------------------------------------------------------

def fetch_top_ids() -> list[int]:
    """
    Scrapes gutenberg.org/browse/scores/top and returns a list of book IDs
    in approximate popularity order (most popular first).
    The page has several sections (top 100 yesterday / 7 days / 30 days).
    We collect all unique IDs, preserving first-seen order (≈ 30-day rank).
    """
    print("Fetching popularity rankings from gutenberg.org…")
    r = requests.get(TOP_BOOKS_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    seen = set()
    ordered = []

    for a in soup.select("ol li a[href]"):
        href = a["href"]
        m = re.search(r"/ebooks/(\d+)", href)
        if m:
            gid = int(m.group(1))
            if gid not in seen:
                seen.add(gid)
                ordered.append(gid)

    print(f"  Found {len(ordered)} ranked book IDs")
    return ordered


# ---------------------------------------------------------------------------
# Step 3: Download a single book as plain text
# ---------------------------------------------------------------------------

CANDIDATE_URLS = [
    # Preferred plain-text UTF-8 cache path
    "{mirror}/cache/epub/{id}/pg{id}.txt",
    # Older naming convention
    "{mirror}/files/{id}/{id}-0.txt",
    "{mirror}/files/{id}/{id}.txt",
]

def download_text(gid: int) -> str | None:
    """Try several URL patterns; return raw text or None on failure."""
    for pattern in CANDIDATE_URLS:
        url = pattern.format(mirror=MIRROR, id=gid)
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and len(r.text) > 5000:
                return r.text
        except requests.RequestException:
            continue
    return None


# ---------------------------------------------------------------------------
# Step 4: Orchestrate
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "filename", "genre", "id", "title", "author",
    "year", "subjects", "locc", "source_url",
]

def year_from_issued(issued: str) -> str:
    m = re.search(r"\b(\d{4})\b", issued)
    return m.group(1) if m else ""


def main() -> None:
    os.makedirs(FICTION_DIR, exist_ok=True)
    os.makedirs(NONFICTION_DIR, exist_ok=True)

    catalog   = fetch_catalog()
    top_ids   = fetch_top_ids()

    counts    = {"fiction": 0, "nonfiction": 0}
    meta_rows = []

    print(f"\nDownloading up to {TARGET_PER_GENRE} fiction + "
          f"{TARGET_PER_GENRE} nonfiction books…\n")

    for gid in top_ids:
        # Stop once both genres are full
        if all(v >= TARGET_PER_GENRE for v in counts.values()):
            break

        if gid not in catalog:
            continue

        meta  = catalog[gid]
        genre = classify(meta["locc"], meta["subjects"])

        if genre == "skip":
            continue
        if counts[genre] >= TARGET_PER_GENRE:
            continue

        print(f"  [{genre.upper():>10}] ({counts[genre]+1:2}/{TARGET_PER_GENRE}) "
              f"#{gid} — {meta['title'][:60]}")

        text = download_text(gid)
        if text is None:
            print(f"    ✗ Could not download #{gid} — skipping")
            continue

        # Save the file
        safe_title = re.sub(r"[^\w\s-]", "", meta["title"])[:50].strip()
        safe_title = re.sub(r"\s+", "_", safe_title)
        filename   = f"{gid}_{safe_title}.txt"
        dest_dir   = FICTION_DIR if genre == "fiction" else NONFICTION_DIR
        filepath   = os.path.join(dest_dir, filename)

        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)

        counts[genre] += 1
        meta_rows.append({
            "filename":   os.path.join(genre, filename),
            "genre":      genre,
            "id":         gid,
            "title":      meta["title"],
            "author":     meta["author"],
            "year":       year_from_issued(meta["issued"]),
            "subjects":   meta["subjects"],
            "locc":       meta["locc"],
            "source_url": f"https://www.gutenberg.org/ebooks/{gid}",
        })

        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    # Balance genres: trim the larger genre down to match the smaller one
    final_count = min(counts["fiction"], counts["nonfiction"])
    if counts["fiction"] != counts["nonfiction"]:
        surplus_genre = "fiction" if counts["fiction"] > counts["nonfiction"] else "nonfiction"
        surplus_dir   = FICTION_DIR if surplus_genre == "fiction" else NONFICTION_DIR
        n_to_remove   = abs(counts["fiction"] - counts["nonfiction"])

        # meta_rows for the surplus genre are already in download order;
        # drop the last n_to_remove of them and delete the files on disk
        surplus_rows    = [r for r in meta_rows if r["genre"] == surplus_genre]
        rows_to_delete  = surplus_rows[-n_to_remove:]
        filenames_to_delete = {r["filename"] for r in rows_to_delete}

        for row in rows_to_delete:
            filepath = os.path.join(OUTPUT_DIR, row["filename"])
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"  Removed (balancing): {row['filename']}")

        meta_rows = [r for r in meta_rows if r["filename"] not in filenames_to_delete]
        print(f"\n  Trimmed {n_to_remove} {surplus_genre} file(s) to balance corpus.")

    # Write metadata CSV
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\n{'='*60}")
    print(f"  Fiction in corpus:     {final_count}")
    print(f"  Nonfiction in corpus:  {final_count}")
    print(f"  Total texts:           {final_count * 2}")
    print(f"  Metadata written to:   {METADATA_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()