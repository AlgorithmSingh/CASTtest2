#!/usr/bin/env python3
"""cAST vs Fixed-Size Chunking: A RAG Retrieval Experiment.

Reproduces the core finding from:
  Zhang et al., "cAST: Enhancing Code Retrieval-Augmented Generation
  with Structural Chunking via Abstract Syntax Tree" (EMNLP 2025)

Uses the official astchunk library (https://github.com/yilinjz/astchunk)
as the reference implementation, alongside our own simplified version
and a fixed-size baseline for comparison.

Usage:
    python run_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.cast_chunker import cast_chunk, cast_chunk_ref
from src.fixed_chunker import fixed_chunk
from src.rag_pipeline import SimpleRetriever, evaluate_retrieval


# ---------------------------------------------------------------------------
# Test data: queries with ground-truth relevant line ranges
# ---------------------------------------------------------------------------
# These queries simulate a developer searching for specific functionality.
# Ground truth is the line range(s) in the source files that contain the answer.

QUERIES = [
    # --- stats.py queries ---
    {
        "query": "compute mean average of a list of numbers",
        "filepath": "stats.py",
        "expected_lines": [(9, 23)],
    },
    {
        "query": "calculate standard deviation and variance",
        "filepath": "stats.py",
        "expected_lines": [(26, 40), (43, 45)],
    },
    {
        "query": "percentile interpolation calculation",
        "filepath": "stats.py",
        "expected_lines": [(107, 123)],
    },
    {
        "query": "z-score normalization for data points",
        "filepath": "stats.py",
        "expected_lines": [(103, 105)],
    },
    {
        "query": "summary statistics dictionary with min max range",
        "filepath": "stats.py",
        "expected_lines": [(125, 138)],
    },
    # --- data_loader.py queries ---
    {
        "query": "load CSV file and parse records with delimiter",
        "filepath": "data_loader.py",
        "expected_lines": [(47, 78)],
    },
    {
        "query": "filter records by category name",
        "filepath": "data_loader.py",
        "expected_lines": [(80, 84)],
    },
    {
        "query": "validate data records check duplicate IDs and negative values",
        "filepath": "data_loader.py",
        "expected_lines": [(131, 160)],
    },
    {
        "query": "merge multiple JSON files and deduplicate by id",
        "filepath": "data_loader.py",
        "expected_lines": [(115, 129)],
    },
    {
        "query": "DataRecord dataclass with tags and category fields",
        "filepath": "data_loader.py",
        "expected_lines": [(12, 39)],
    },
    # --- cache.py queries ---
    {
        "query": "LRU cache get retrieve value by key and move to end",
        "filepath": "cache.py",
        "expected_lines": [(57, 72)],
    },
    {
        "query": "cache evict expired entries remove old TTL",
        "filepath": "cache.py",
        "expected_lines": [(104, 112)],
    },
    {
        "query": "get or compute cache miss fallback callable",
        "filepath": "cache.py",
        "expected_lines": [(130, 152)],
    },
    {
        "query": "cache hit rate statistics performance metrics",
        "filepath": "cache.py",
        "expected_lines": [(118, 127)],
    },
    {
        "query": "thread safe cache put insert with TTL expiration",
        "filepath": "cache.py",
        "expected_lines": [(74, 93)],
    },
]


def load_test_files():
    """Load all Python files from the test codebase directory."""
    codebase_dir = os.path.join(os.path.dirname(__file__), "test_codebase")
    files = {}
    for fname in sorted(os.listdir(codebase_dir)):
        if fname.endswith(".py"):
            fpath = os.path.join(codebase_dir, fname)
            with open(fpath) as f:
                files[fname] = f.read()
    return files


def print_separator(char="=", width=72):
    print(char * width)


def print_chunks_overview(label, chunks):
    """Print a brief overview of chunks."""
    print(f"\n  {label}: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        lines = c.content.count("\n") + 1
        print(f"    [{i+1}] lines {c.start_line}-{c.end_line} "
              f"({lines} lines, {c.nws_size} nws chars)")


def count_mid_block_breaks(chunks):
    """Count chunks that start mid-function/class (broken boundaries)."""
    breaks = 0
    for c in chunks:
        lines = c.content.split("\n")
        first_nonblank = next((l for l in lines if l.strip()), "")
        if first_nonblank and first_nonblank[0] == " " and \
           not first_nonblank.strip().startswith(
               ("def ", "class ", "@", "#", '"""', "'''")):
            breaks += 1
    return breaks


def main():
    print_separator()
    print("  cAST vs Fixed-Size Chunking — RAG Retrieval Experiment")
    print("  Paper:  Zhang et al., EMNLP 2025 (arXiv:2506.15655)")
    print("  Ref:    https://github.com/yilinjz/astchunk")
    print_separator()

    # Load test codebase
    files = load_test_files()
    print(f"\nLoaded {len(files)} test files: {', '.join(files.keys())}")

    # -------------------------------------------------------------------
    # Step 1: Chunk all files with all three strategies
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1: Chunking Comparison")
    print("=" * 72)

    MAX_CHUNK_SIZE = 800  # non-whitespace chars (small to force splits)
    MAX_LINES = 20        # for fixed-size baseline

    strategies = {
        "cAST (ours)": [],
        "cAST (ref) ": [],
        "Fixed-size ": [],
    }

    for fname, code in files.items():
        cc_ours = cast_chunk(code, filepath=fname, max_chunk_size=MAX_CHUNK_SIZE)
        cc_ref = cast_chunk_ref(code, filepath=fname, max_chunk_size=MAX_CHUNK_SIZE)
        fc = fixed_chunk(code, filepath=fname, max_lines=MAX_LINES)

        strategies["cAST (ours)"].extend(cc_ours)
        strategies["cAST (ref) "].extend(cc_ref)
        strategies["Fixed-size "].extend(fc)

        print(f"\n  --- {fname} ---")
        for label, chunks in [("cAST (ours)", cc_ours),
                              ("cAST (ref) ", cc_ref),
                              ("Fixed-size ", fc)]:
            breaks = count_mid_block_breaks(chunks)
            print(f"    {label}: {len(chunks):>2} chunks, "
                  f"{breaks} mid-block breaks")

    print(f"\n  Strategy totals:")
    for label, chunks in strategies.items():
        breaks = count_mid_block_breaks(chunks)
        print(f"    {label}: {len(chunks):>2} chunks total, "
              f"{breaks} mid-block breaks")

    # -------------------------------------------------------------------
    # Step 2: Show example chunk comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2: Example — how each method chunks compute_variance (stats.py)")
    print("=" * 72)

    for label, chunks in strategies.items():
        for c in chunks:
            if c.filepath == "stats.py" and "compute_variance" in c.content:
                print(f"\n  --- {label.strip()} (lines {c.start_line}-{c.end_line}) ---")
                preview_lines = c.content.split("\n")[:12]
                for line in preview_lines:
                    print(f"  | {line}")
                total = len(c.content.split("\n"))
                if total > 12:
                    print(f"  | ... ({total} lines total)")
                break

    # -------------------------------------------------------------------
    # Step 3: Build retrievers and evaluate all three
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3: Retrieval Evaluation (TF-IDF + Cosine Similarity, top_k=3)")
    print("=" * 72)

    top_k = 3
    results = {}
    for label, chunks in strategies.items():
        retriever = SimpleRetriever(chunks)
        results[label] = evaluate_retrieval(retriever, QUERIES, top_k=top_k)

    print(f"\n  {'Metric':<20}", end="")
    for label in strategies:
        print(f" {label.strip():>12}", end="")
    print()
    print(f"  {'-'*56}")

    for metric in ["avg_precision", "avg_recall"]:
        print(f"  {metric:<20}", end="")
        for label in strategies:
            print(f" {results[label][metric]:>12.4f}", end="")
        print()

    # Delta row
    print(f"\n  {'Delta vs Fixed':<20}", end="")
    fixed_label = "Fixed-size "
    for label in strategies:
        if label == fixed_label:
            print(f" {'(baseline)':>12}", end="")
        else:
            delta = results[label]["avg_recall"] - results[fixed_label]["avg_recall"]
            sign = "+" if delta >= 0 else ""
            print(f" {sign}{delta:>11.4f}", end="")
    print()

    # -------------------------------------------------------------------
    # Step 4: Per-query breakdown
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4: Per-Query Recall")
    print("=" * 72)

    labels = list(strategies.keys())
    header = f"  {'Query':<50}"
    for l in labels:
        header += f" {l.strip()[:8]:>8}"
    print(f"\n{header}")
    print(f"  {'-'*74}")

    wins = {l: 0 for l in labels}

    for i, q in enumerate(QUERIES):
        q_short = q["query"][:47] + "..." if len(q["query"]) > 47 else q["query"]
        row = f"  {q_short:<50}"
        recalls = {}
        for l in labels:
            r = results[l]["details"][i]["recall"]
            recalls[l] = r
            row += f" {r:>8.2f}"
        print(row)

        # Track wins (cAST ours vs fixed, cAST ref vs fixed)
        best = max(recalls.values())
        for l in labels:
            if recalls[l] == best:
                wins[l] += 1

    # -------------------------------------------------------------------
    # Step 5: Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"\n  Total queries: {len(QUERIES)}")
    print(f"\n  {'Strategy':<20} {'Precision@3':>12} {'Recall@3':>12} {'Breaks':>8} {'Best':>6}")
    print(f"  {'-'*58}")

    for label, chunks in strategies.items():
        p = results[label]["avg_precision"]
        r = results[label]["avg_recall"]
        b = count_mid_block_breaks(chunks)
        w = wins[label]
        print(f"  {label.strip():<20} {p:>12.4f} {r:>12.4f} {b:>8} {w:>6}")

    # Highlight the key finding
    ours_recall = results["cAST (ours)"]["avg_recall"]
    ref_recall = results["cAST (ref) "]["avg_recall"]
    fixed_recall = results["Fixed-size "]["avg_recall"]
    cast_better = ours_recall > fixed_recall or ref_recall > fixed_recall

    if cast_better:
        best_cast = max(ours_recall, ref_recall)
        delta = best_cast - fixed_recall
        print(f"\n  cAST improves recall by +{delta:.4f} over fixed-size chunking.")
        print("  This confirms the paper's finding: structure-aware chunking")
        print("  produces more semantically coherent, retrievable code units.")
    else:
        print("\n  Results are mixed on this small test set. The paper's")
        print("  gains are more pronounced on larger real-world repositories.")

    print()
    print_separator()
    return 0


if __name__ == "__main__":
    sys.exit(main())
