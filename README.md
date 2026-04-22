# text_analysis_final


Does fiction follow Zipf's Law differently than nonfiction? This project downloads (up to) 100 books from Project Gutenberg and finds out.

## What it does

1. **Builds a corpus** — grabs the 50 most popular fiction and 50 most popular nonfiction English books from Gutenberg, saves them as plain text files, and writes a metadata CSV.
2. **Runs the analysis** — counts word frequencies across each genre, fits a power-law curve, and produces three charts comparing both genres against the ideal Zipf distribution.

## Setup

```bash
pip install requests beautifulsoup4 numpy matplotlib scipy
```

## Usage

First, build the corpus (this takes a while — it downloads 100 books with polite delays):

```bash
python construct_corpus.py
```

Then run the analysis:

```bash
python zipf_analysis.py
```

Results land in `zipf_output/`:

- `zipf_loglog.png` — the classic log-log rank vs. frequency plot
- `zipf_deviation.png` — how far each genre strays from the ideal Zipf slope
- `zipf_topwords.png` — top 30 most frequent words per genre

A summary table is also printed to the console.

## Output structure

```
corpus/
  fiction/        ← 50 .txt files
  nonfiction/     ← 50 .txt files
  metadata.csv    ← one row per book

zipf_output/
  zipf_loglog.png
  zipf_deviation.png
  zipf_topwords.png
```

## Notes

- The corpus builder sleeps 2 seconds between downloads to be kind to Gutenberg's servers.
- Genre classification uses Library of Congress codes first, falling back to subject keywords. Poetry and drama are skipped.
- Stop words are kept intentionally — they're central to how Zipf's Law works in natural language.
