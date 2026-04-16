"""
zipf_analysis.py

Compares fiction vs. nonfiction texts from the Gutenberg corpus against Zipf's Law.

What is Zipf's Law?
    The r-th most frequent word appears roughly 1/r as often as the most frequent.
    Equivalently, on a log-log plot of rank vs. frequency, the data should fall
    on a straight line with slope ≈ -1.

This script:
    1. Reads metadata.csv to find fiction / nonfiction files.
    2. Combines all text in each genre (stripping Gutenberg boilerplate).
    3. Counts word frequencies.
    4. Fits a power-law (linear regression in log-log space) to each genre.
    5. Produces three publication-quality figures:
          zipf_loglog.png        — log-log rank vs frequency, both genres + ideal
          zipf_deviation.png     — per-rank deviation from ideal Zipf slope
          zipf_topwords.png      — top-30 word frequencies as a bar chart

Usage:
    pip install numpy matplotlib scipy
    python zipf_analysis.py [--corpus ./corpus] [--metadata ./corpus/metadata.csv]

All non-alphabetic tokens are dropped; case is folded to lowercase.
Stop words are intentionally KEPT — they are the backbone of Zipf's Law.
"""

import argparse
import csv
import os
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_DIR   = "./corpus"
METADATA_CSV = "./corpus/metadata.csv"
OUTPUT_DIR   = "./zipf_output"

# Gutenberg boilerplate markers
START_MARKER = re.compile(r"\*{3}\s*START OF THE PROJECT GUTENBERG.+?\*{3}", re.IGNORECASE)
END_MARKER   = re.compile(r"\*{3}\s*END OF THE PROJECT GUTENBERG.+?\*{3}",   re.IGNORECASE)

# Plotting style
COLORS = {"fiction": "#E8593C", "nonfiction": "#2D7DD2"}
IDEAL_COLOR = "#888888"

# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def strip_boilerplate(text: str) -> str:
    """Remove Gutenberg header and footer, returning only the book body."""
    start = START_MARKER.search(text)
    end   = END_MARKER.search(text)
    body_start = start.end() if start else 0
    body_end   = end.start() if end   else len(text)
    return text[body_start:body_end]


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens only (no punctuation, no numbers)."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def load_genre(genre: str, metadata_path: str, corpus_dir: str) -> Counter:
    """Read all texts for a genre, strip boilerplate, tokenize, count."""
    counter: Counter = Counter()
    n_files = 0

    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genre"].strip().lower() != genre:
                continue
            filepath = os.path.join(corpus_dir, row["filename"])
            if not os.path.exists(filepath):
                print(f"  ⚠  Missing file: {filepath}")
                continue
            with open(filepath, encoding="utf-8", errors="replace") as tf:
                raw = tf.read()
            body   = strip_boilerplate(raw)
            tokens = tokenize(body)
            counter.update(tokens)
            n_files += 1

    print(f"  {genre.capitalize():>12}: {n_files} files, "
          f"{sum(counter.values()):,} total tokens, "
          f"{len(counter):,} unique words")
    return counter


# ---------------------------------------------------------------------------
# Zipf analysis
# ---------------------------------------------------------------------------

def zipf_arrays(counter: Counter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        ranks      — 1-based integer ranks
        freqs      — raw word counts, sorted descending
        rel_freqs  — freqs / total  (relative frequency)
    """
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    return ranks, freqs, freqs / freqs.sum()


def fit_powerlaw(ranks: np.ndarray, freqs: np.ndarray,
                 max_rank: int = 5000) -> tuple[float, float, float]:
    """
    OLS regression of log(freq) ~ log(rank) over the top `max_rank` words.
    Returns (slope, intercept, r_squared).
    Zipf's ideal: slope ≈ -1.
    """
    log_r = np.log10(ranks[:max_rank])
    log_f = np.log10(freqs[:max_rank])
    slope, intercept, r, *_ = stats.linregress(log_r, log_f)
    return slope, intercept, r ** 2


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#FAFAFA",
        "axes.grid":         True,
        "grid.color":        "#E0E0E0",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom":True,
        "axes.labelsize":    12,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "legend.framealpha": 0.9,
        "font.family":       "serif",
    })


# ---------------------------------------------------------------------------
# Figure 1: Log-log rank vs frequency
# ---------------------------------------------------------------------------

def plot_loglog(data: dict, output_path: str) -> None:
    """
    Main Zipf plot.  Each genre is drawn as a scatter of the top 10 000 words,
    with the fitted power-law line and the ideal Zipf line overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    max_plot = 10_000

    for genre, (ranks, freqs, rel_freqs, slope, intercept, r2) in data.items():
        color = COLORS[genre]
        n = min(max_plot, len(ranks))

        # Scatter (thin, semi-transparent so overlap is visible)
        ax.scatter(ranks[:n], rel_freqs[:n],
                   s=2, alpha=0.25, color=color, rasterized=True)

        # Fitted power-law line
        fit_x = np.logspace(0, np.log10(ranks[n - 1]), 300)
        fit_y = (10 ** intercept) * fit_x ** slope / freqs.sum()
        ax.plot(fit_x, fit_y, color=color, linewidth=2.2,
                label=f"{genre.capitalize()}  (slope = {slope:.3f}, R²={r2:.4f})")

    # Ideal Zipf line: f ∝ 1/rank, normalised to pass through fiction's rank-1
    fiction_ranks, fiction_freqs, fiction_rel, *_ = data["fiction"]
    ideal_y0 = fiction_rel[0]
    ideal_x  = np.logspace(0, np.log10(max_plot), 300)
    ideal_y  = ideal_y0 / ideal_x          # slope = -1 exactly
    ax.plot(ideal_x, ideal_y, color=IDEAL_COLOR, linewidth=1.5,
            linestyle="--", label="Ideal Zipf  (slope = −1.000)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank", labelpad=8)
    ax.set_ylabel("Relative frequency", labelpad=8)
    ax.set_title("Word Frequency vs. Rank — Zipf's Law\n"
                 "Fiction vs. Nonfiction (Project Gutenberg corpus)")
    ax.legend(loc="upper right")

    # Annotate the slope comparison
    slope_f = data["fiction"][3]
    slope_n = data["nonfiction"][3]
    note = (f"Δ slope = {abs(slope_f - slope_n):.3f}\n"
            f"(ideal = −1.000)")
    ax.text(0.03, 0.07, note, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Deviation from ideal Zipf slope
# ---------------------------------------------------------------------------

def plot_deviation(data: dict, output_path: str) -> None:
    """
    For each genre, compute the residual of log10(freq) against the ideal
    Zipf line (slope = -1) anchored at rank 1.  Positive = more common than
    Zipf predicts; negative = less common.
    Smoothed with a rolling median for readability.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    max_plot = 5_000
    window   = 50   # rolling median window

    for ax, (genre, (ranks, freqs, rel_freqs, slope, intercept, r2)) in \
            zip(axes, data.items()):
        color = COLORS[genre]
        n = min(max_plot, len(ranks))

        # Ideal prediction anchored at rank 1
        ideal_log = np.log10(rel_freqs[0]) + (-1) * (np.log10(ranks[:n]) - 0)
        actual_log = np.log10(rel_freqs[:n])
        residual   = actual_log - ideal_log

        # Rolling median
        pad = window // 2
        smoothed = np.array([
            np.median(residual[max(0, i - pad): i + pad + 1])
            for i in range(n)
        ])

        ax.axhline(0, color=IDEAL_COLOR, linewidth=1.2, linestyle="--",
                   label="Ideal Zipf (residual = 0)")
        ax.fill_between(ranks[:n], residual, alpha=0.12, color=color)
        ax.plot(ranks[:n], smoothed, color=color, linewidth=1.8,
                label=f"Smoothed residual (window={window})")
        ax.scatter(ranks[:n], residual, s=1, alpha=0.15,
                   color=color, rasterized=True)

        ax.set_xscale("log")
        ax.set_xlabel("Rank")
        ax.set_title(f"{genre.capitalize()}\nDeviation from Ideal Zipf")
        ax.legend(fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("log₁₀(actual freq) − log₁₀(Zipf prediction)")

    fig.suptitle("Per-Rank Deviation from Zipf's Law", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Top-30 words bar chart
# ---------------------------------------------------------------------------

def plot_topwords(counters: dict[str, Counter], output_path: str) -> None:
    """
    Horizontal bar chart of the 30 most frequent words in each genre,
    showing relative frequency (%).  Displayed side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))

    for ax, (genre, counter) in zip(axes, counters.items()):
        color = COLORS[genre]
        total = sum(counter.values())
        top30 = counter.most_common(30)
        words  = [w for w, _ in reversed(top30)]
        pct    = [c / total * 100 for _, c in reversed(top30)]

        bars = ax.barh(words, pct, color=color, alpha=0.82, edgecolor="white")

        # Value labels
        for bar, p in zip(bars, pct):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{p:.3f}%", va="center", fontsize=8)

        ax.set_xlabel("Relative frequency (%)")
        ax.set_title(f"{genre.capitalize()}\nTop 30 Words")
        ax.set_xlim(right=max(pct) * 1.22)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Most Frequent Words by Genre", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(data: dict, counters: dict) -> None:
    print("\n" + "=" * 64)
    print(f"{'ZIPF\'S LAW ANALYSIS SUMMARY':^64}")
    print("=" * 64)
    print(f"{'Metric':<32} {'Fiction':>14} {'Nonfiction':>14}")
    print("-" * 64)

    labels = ["Fitted slope", "Intercept (log₁₀)", "R² (fit quality)",
              "Total tokens", "Unique words", "Vocab richness (TTR %)"]
    for genre, (ranks, freqs, rel_freqs, slope, intercept, r2) in data.items():
        pass   # just iterating to confirm order

    f  = data["fiction"]
    nf = data["nonfiction"]
    cf = counters["fiction"]
    cn = counters["nonfiction"]

    rows = [
        (f[3],  nf[3],  "{:.4f}"),
        (f[4],  nf[4],  "{:.4f}"),
        (f[5],  nf[5],  "{:.4f}"),
        (sum(cf.values()),  sum(cn.values()),  "{:,.0f}"),
        (len(cf),           len(cn),           "{:,.0f}"),
        (len(cf)/sum(cf.values())*100, len(cn)/sum(cn.values())*100, "{:.4f}"),
    ]
    for label, (fv, nv, fmt) in zip(labels, rows):
        print(f"  {label:<30} {fmt.format(fv):>14} {fmt.format(nv):>14}")

    print("-" * 64)
    print(f"  Ideal Zipf slope: -1.0000")
    print(f"  Fiction  deviation from ideal: {abs(f[3] - (-1)):.4f}")
    print(f"  Nonfiction deviation from ideal: {abs(nf[3] - (-1)):.4f}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Zipf's Law analysis on Gutenberg corpus")
    parser.add_argument("--corpus",   default=CORPUS_DIR,   help="Corpus root directory")
    parser.add_argument("--metadata", default=METADATA_CSV, help="Path to metadata.csv")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    print("Loading corpus…")
    counters = {}
    for genre in ("fiction", "nonfiction"):
        counters[genre] = load_genre(genre, args.metadata, args.corpus)

    print("\nFitting power laws…")
    data = {}
    for genre, counter in counters.items():
        ranks, freqs, rel_freqs = zipf_arrays(counter)
        slope, intercept, r2   = fit_powerlaw(ranks, freqs)
        data[genre]             = (ranks, freqs, rel_freqs, slope, intercept, r2)
        print(f"  {genre.capitalize():>12}: slope={slope:.4f}, "
              f"intercept={intercept:.4f}, R²={r2:.4f}")

    print("\nGenerating figures…")
    plot_loglog(  data,     os.path.join(OUTPUT_DIR, "zipf_loglog.png"))
    plot_deviation(data,    os.path.join(OUTPUT_DIR, "zipf_deviation.png"))
    plot_topwords(counters, os.path.join(OUTPUT_DIR, "zipf_topwords.png"))

    print_summary(data, counters)

    print(f"\nAll output written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()